"""End-to-end tests for the s2mirror package: synthetic S2AG dump files
-> mapper partitions -> reducer shard DBs -> store -> FastAPI server."""

from __future__ import annotations

import gzip
import json

import pytest
from fastapi.testclient import TestClient

from s2mirror import mapper, reducer, schema
from s2mirror.server import create_app
from s2mirror.store import MirrorStore

SHA1 = "a" * 39 + "1"
SHA2 = "b" * 39 + "2"
SHA3 = "c" * 39 + "3"
SHA1_ALIAS = "d" * 39 + "4"

PAPERS = [
    {
        "corpusid": 101,
        "externalids": {"DOI": "10.1000/XYZ", "CorpusId": "101", "ArXiv": None},
        "url": f"https://www.semanticscholar.org/paper/{SHA1}",
        "title": "Alpha methods paper",
        "authors": [{"authorId": "7001", "name": "Ada One"}],
        "venue": "Nature Methods",
        "year": 2020,
        "referencecount": 0,
        "citationcount": 2,
        "influentialcitationcount": 1,
        "isopenaccess": True,
        "s2fieldsofstudy": [{"category": "Biology", "source": "s2-fos-model"}],
        "publicationtypes": ["JournalArticle"],
        "publicationdate": "2020-05-01",
        "journal": {"name": "Nat. Methods"},
    },
    {
        "corpusid": 202,
        "externalids": {"ArXiv": "2005.11687", "CorpusId": "202"},
        "url": f"https://www.semanticscholar.org/paper/{SHA2}",
        "title": "Beta statistics paper",
        "authors": [
            {"authorId": "7001", "name": "Ada One"},
            {"authorId": "7002", "name": "Bo Two"},
        ],
        "venue": "arXiv.org",
        "year": 2021,
        "referencecount": 2,
        "citationcount": 1,
        "influentialcitationcount": 0,
        "isopenaccess": False,
        "s2fieldsofstudy": None,
        "publicationtypes": None,
        "publicationdate": None,
        "journal": None,
    },
    {
        "corpusid": 303,
        "externalids": {"CorpusId": "303"},
        "url": f"https://www.semanticscholar.org/paper/{SHA3}",
        "title": "Gamma survey",
        "authors": [],
        "venue": "",
        "year": 2022,
        "referencecount": 2,
        "citationcount": 0,
        "influentialcitationcount": 0,
        "isopenaccess": False,
        "s2fieldsofstudy": None,
        "publicationtypes": ["Review"],
        "publicationdate": "2022-01-15",
        "journal": None,
    },
]

ABSTRACTS = [
    {
        "corpusid": 101,
        "openaccessinfo": {"url": "https://arxiv.org/pdf/1", "status": "GREEN"},
        "abstract": "We introduce the alpha method.",
    },
]

CITATIONS = [
    {"citingcorpusid": 202, "citedcorpusid": 101, "isinfluential": True,
     "intents": ["methodology"], "contexts": ["as shown in [1]"]},
    {"citingcorpusid": 303, "citedcorpusid": 101, "isinfluential": False,
     "intents": None, "contexts": None},
    {"citingcorpusid": 303, "citedcorpusid": 202, "isinfluential": False,
     "intents": ["background"], "contexts": None},
    # dangling target: no papers-dataset row for 999
    {"citingcorpusid": 202, "citedcorpusid": 999, "isinfluential": False,
     "intents": None, "contexts": None},
    # the S2AG dump materializes most edges twice (two shard families) —
    # exact twin of the 303->101 edge; the reducer must collapse it
    {"citingcorpusid": 303, "citedcorpusid": 101, "isinfluential": False,
     "intents": None, "contexts": None},
]

PAPER_IDS = [
    {"sha": SHA1, "corpusid": 101, "primary": True},
    {"sha": SHA1_ALIAS, "corpusid": 101, "primary": False},
    {"sha": SHA2, "corpusid": 202, "primary": True},
    {"sha": SHA3, "corpusid": 303, "primary": True},
]

AUTHORS = [
    {"authorid": "7001", "name": "Ada One", "aliases": ["A. One"],
     "affiliations": ["MIT"], "homepage": None, "papercount": 2,
     "citationcount": 30, "hindex": 5, "externalids": None,
     "url": "https://www.semanticscholar.org/author/7001"},
    {"authorid": "7002", "name": "Bo Two", "aliases": None, "affiliations": None,
     "homepage": None, "papercount": 1, "citationcount": 4, "hindex": 1,
     "externalids": None, "url": "https://www.semanticscholar.org/author/7002"},
]


def _write_dump(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    base = tmp_path_factory.mktemp("s2mirror")
    dumps, parts, store, scratch = (base / d for d in ("dumps", "parts", "store", "scratch"))
    for d in (dumps, parts, store, scratch):
        d.mkdir()
    datasets = {
        "papers": PAPERS, "abstracts": ABSTRACTS, "citations": CITATIONS,
        "paper-ids": PAPER_IDS, "authors": AUTHORS,
    }
    for name, rows in datasets.items():
        f = dumps / f"{name}.jsonl.gz"
        _write_dump(f, rows)
        mapper.map_dataset_file(name, f, parts, 0, scratch=scratch)
    for kind, (_fn, n_shards) in reducer.REDUCERS.items():
        for shard in range(n_shards):
            reducer.reduce_shard(kind, shard, parts, store, scratch=scratch)
    return store


@pytest.fixture(scope="module")
def store(store_root):
    return MirrorStore(store_root)


class TestStore:
    def test_resolve_forms(self, store):
        assert store.resolve(SHA1) == 101
        assert store.resolve(SHA1_ALIAS) == 101          # merged-alias sha
        assert store.resolve("DOI:10.1000/xyz") == 101   # case-normalized
        assert store.resolve("DOI:10.1000/XYZ") == 101
        assert store.resolve("ARXIV:2005.11687") == 202
        # DataCite arXiv DOI normalizes to the arXiv id
        assert store.resolve("DOI:10.48550/arXiv.2005.11687") == 202
        assert store.resolve("CorpusId:303") == 303
        assert store.resolve("nope") is None
        assert store.resolve("DOI:10.9999/none") is None

    def test_resolve_many_matches_resolve(self, store):
        raws = [SHA1, SHA1_ALIAS, "DOI:10.1000/XYZ", "ARXIV:2005.11687",
                "CorpusId:303", "nope", "DOI:10.9999/none", SHA1]
        many = store.resolve_many(raws)
        for r in raws:
            assert many[r] == store.resolve(r), r

    def test_paper_record_shape(self, store):
        rec = store.get_paper(101)
        assert rec["paperId"] == SHA1
        assert rec["title"] == "Alpha methods paper"
        assert rec["abstract"] == "We introduce the alpha method."
        assert rec["openAccessPdf"] == {"url": "https://arxiv.org/pdf/1", "status": "GREEN"}
        assert rec["externalIds"]["DOI"] == "10.1000/XYZ"
        assert rec["s2FieldsOfStudy"] == [{"category": "Biology", "source": "s2-fos-model"}]
        assert rec["authors"] == [{"authorId": "7001", "name": "Ada One"}]
        assert store.get_paper(202)["abstract"] is None
        assert store.get_paper(999) is None

    def test_adjacency(self, store):
        citers = store.adjacency("citers", 101)
        # the duplicated 303->101 edge collapses to one entry
        assert [int(r["other"]) for r in citers] == [303, 202]  # newest-first
        infl, intents = schema.unpack_flags(int(citers[1]["flags"]))
        assert infl is True and intents == ["methodology"]
        refs = store.adjacency("refs", 303)
        assert sorted(int(r["other"]) for r in refs) == [101, 202]
        assert len(store.adjacency("refs", 101)) == 0

    def test_read_time_dedupe_of_legacy_blobs(self, store):
        import numpy as np
        from s2mirror.reducer import ADJ_DTYPE
        legacy = np.array([(7, 0), (5, 1), (7, 0), (3, 0), (5, 1)], dtype=ADJ_DTYPE)
        clean = store._dedupe(legacy)
        assert [int(r["other"]) for r in clean] == [7, 5, 3]  # order preserved

    def test_authors(self, store):
        a = store.get_author(7001)
        assert a["name"] == "Ada One" and a["hIndex"] == 5
        assert [int(c) for c in store.author_paper_ids(7001)] == [202, 101]


KEY = "test-mirror-key"


class _FakeUpstream:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload
        self.calls = []

    def request(self, method, path, params=None, json_body=None):
        self.calls.append((method, path, params, json_body))
        return self.status, self.payload


@pytest.fixture()
def client(store):
    app = create_app(store, api_keys={KEY})
    return TestClient(app)


def _get(client, path, **params):
    return client.get(path, params=params, headers={"x-api-key": KEY})


class TestServer:
    def test_auth_required(self, client):
        assert client.get("/graph/v1/paper/" + SHA1).status_code == 401
        assert client.get("/health").status_code == 200  # health is open

    def test_single_paper_by_doi_path(self, client):
        r = _get(client, "/graph/v1/paper/DOI:10.1000/xyz",
                 fields="title,externalIds,citationCount")
        assert r.status_code == 200
        body = r.json()
        assert body["paperId"] == SHA1
        assert body["title"] == "Alpha methods paper"
        assert body["citationCount"] == 2
        assert "venue" not in body  # not requested

    def test_paper_batch_alignment(self, client):
        r = client.post(
            f"/graph/v1/paper/batch?fields=title",
            json={"ids": [SHA1, "CorpusId:202", "DOI:10.404/missing"]},
            headers={"x-api-key": KEY},
        )
        body = r.json()
        assert body[0]["title"] == "Alpha methods paper"
        assert body[1]["title"] == "Beta statistics paper"
        assert body[2] is None

    def test_references_with_edge_fields(self, client):
        r = _get(client, f"/graph/v1/paper/{SHA3}/references",
                 fields="citedPaper.paperId,citedPaper.title,contexts,intents,isInfluential")
        body = r.json()
        assert body["offset"] == 0 and "next" not in body
        titles = {e["citedPaper"]["title"] for e in body["data"]}
        assert titles == {"Alpha methods paper", "Beta statistics paper"}
        for e in body["data"]:
            assert e["contexts"] == [] and isinstance(e["intents"], list)

    def test_citations_pagination(self, client):
        r = _get(client, f"/graph/v1/paper/{SHA1}/citations",
                 fields="citingPaper.paperId,citingPaper.citationCount", limit=1)
        body = r.json()
        assert body["next"] == 1 and len(body["data"]) == 1
        assert body["data"][0]["citingPaper"]["paperId"] == SHA3
        r2 = _get(client, f"/graph/v1/paper/{SHA1}/citations",
                  fields="citingPaper.paperId", limit=1, offset=1)
        body2 = r2.json()
        assert body2["data"][0]["citingPaper"]["paperId"] == SHA2
        assert "next" not in body2

    def test_dangling_edge_target(self, client):
        r = _get(client, f"/graph/v1/paper/{SHA2}/references",
                 fields="citedPaper.paperId,isInfluential")
        ids = [e["citedPaper"]["paperId"] for e in r.json()["data"]]
        assert SHA1 in ids and None in ids  # 999 has no papers row

    def test_author_batch(self, client):
        r = client.post(
            "/graph/v1/author/batch?fields=name,hIndex,paperCount,affiliations",
            json={"ids": ["7001", "7002", "424242"]},
            headers={"x-api-key": KEY},
        )
        body = r.json()
        assert body[0]["name"] == "Ada One" and body[0]["hIndex"] == 5
        assert body[0]["authorId"] == "7001"
        assert body[1]["affiliations"] == []
        assert body[2] is None

    def test_projection_cache_stable_across_shapes(self, client):
        """Same request twice -> identical body (cache hit); a different
        fields shape must not be polluted by the cached projection."""
        p = f"/graph/v1/paper/{SHA3}/references"
        a1 = _get(client, p, fields="citedPaper.paperId,citedPaper.title").json()
        a2 = _get(client, p, fields="citedPaper.paperId,citedPaper.title").json()
        assert a1 == a2
        b = _get(client, p, fields="citedPaper.paperId,citedPaper.year").json()
        inner = b["data"][0]["citedPaper"]
        assert "title" not in inner and "year" in inner

    def test_author_papers(self, client):
        r = _get(client, "/graph/v1/author/7001/papers",
                 fields="paperId,title,year,venue,citationCount")
        body = r.json()
        assert [e["paperId"] for e in body["data"]] == [SHA2, SHA1]
        assert body["data"][0]["venue"] == "arXiv.org"


class TestConcurrency:
    def test_concurrent_mixed_load_all_200(self, store):
        """Regression: a shared sqlite3 connection raced under concurrent
        statements ('bad parameter or other API misuse' -> 500s)."""
        import concurrent.futures as cf
        app = create_app(store, api_keys={KEY})
        client = TestClient(app, raise_server_exceptions=False)
        h = {"x-api-key": KEY}

        def hit(i):
            kind = i % 3
            if kind == 0:
                return client.post("/graph/v1/paper/batch?fields=title,authors.name",
                                   json={"ids": [SHA1, SHA2, SHA3, "CorpusId:101"]},
                                   headers=h).status_code
            if kind == 1:
                return client.get(f"/graph/v1/paper/{SHA1}",
                                  params={"fields": "title,citationCount"},
                                  headers=h).status_code
            return client.get(f"/graph/v1/paper/{SHA1}/citations",
                              params={"fields": "citingPaper.paperId", "limit": 100},
                              headers=h).status_code

        with cf.ThreadPoolExecutor(max_workers=16) as pool:
            codes = list(pool.map(hit, range(600)))
        assert set(codes) == {200}, {c: codes.count(c) for c in set(codes)}


class TestUpstreamFallback:
    def test_unknown_paper_proxied_and_memoized(self, store):
        fake = _FakeUpstream(payload={"paperId": "f" * 40, "title": "Fresh"})
        app = create_app(store, api_keys={KEY}, upstream=fake)
        c = TestClient(app)
        for _ in range(2):
            r = c.get("/graph/v1/paper/DOI:10.5555/fresh",
                      params={"fields": "title"}, headers={"x-api-key": KEY})
            assert r.status_code == 200 and r.json()["title"] == "Fresh"
        assert len(fake.calls) == 1  # memo served the second hit

    def test_unknown_paper_404_without_upstream(self, store):
        c = TestClient(create_app(store, api_keys={KEY}))
        r = c.get("/graph/v1/paper/DOI:10.5555/fresh", headers={"x-api-key": KEY})
        assert r.status_code == 404

    def test_embedding_fields_proxied(self, store):
        fake = _FakeUpstream(payload=[{"paperId": SHA1, "embedding": {"vector": [0.1]}}])
        app = create_app(store, api_keys={KEY}, upstream=fake)
        c = TestClient(app)
        r = c.post("/graph/v1/paper/batch?fields=paperId,embedding.specter_v2",
                   json={"ids": [SHA1]}, headers={"x-api-key": KEY})
        assert r.json()[0]["embedding"]["vector"] == [0.1]
        assert fake.calls and fake.calls[0][1] == "/graph/v1/paper/batch"

    def test_batch_misses_subbatched_upstream(self, store):
        fake = _FakeUpstream(payload=[{"paperId": "e" * 40, "title": "New paper"}])
        app = create_app(store, api_keys={KEY}, upstream=fake)
        c = TestClient(app)
        r = c.post("/graph/v1/paper/batch?fields=title",
                   json={"ids": [SHA1, "DOI:10.777/new"]}, headers={"x-api-key": KEY})
        body = r.json()
        assert body[0]["title"] == "Alpha methods paper"
        assert body[1]["title"] == "New paper"
        # only the miss went upstream
        assert fake.calls[0][3] == {"ids": ["DOI:10.777/new"]}

    def test_search_passthrough(self, store):
        fake = _FakeUpstream(payload={"data": [], "total": 0})
        app = create_app(store, api_keys={KEY}, upstream=fake)
        c = TestClient(app)
        r = c.get("/graph/v1/paper/search/bulk", params={"query": "alpha"},
                  headers={"x-api-key": KEY})
        assert r.status_code == 200
        assert fake.calls[0][1] == "/graph/v1/paper/search/bulk"
        assert fake.calls[0][2] == {"query": "alpha"}
