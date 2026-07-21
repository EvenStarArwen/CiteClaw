"""End-to-end + unit tests for the S2ORC full-text mirror library.

Everything here runs offline (no Modal, no network): synthetic dump
records are mapped, reduced into real on-disk SQLite shards, and read
back through the store and the FastAPI surface. Run with PYTHONPATH=src.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from s2orc_mirror import mapper, reducer, schema
from s2orc_mirror.store import MirrorStore

# ---- synthetic corpus ------------------------------------------------------

RECORDS = [
    {
        "corpusid": 89669906,
        "externalids": {"doi": "10.5935/2177-4560.20110018", "arxiv": None,
                        "pubmed": "12345", "pubmedcentral": None, "mag": "999", "acl": None},
        "content": {
            "source": {"pdfurls": None, "pdfsha": "abc",
                       "oainfo": {"license": "CCBY", "status": "GOLD",
                                  "openaccessurl": "https://ex.org/one.pdf"}},
            "text": "Body one about proteins and folding.",
            "annotations": {"paragraph": '[{"start":0,"end":10}]',
                            "sectionheader": '[{"start":0,"end":4}]'},
        },
    },
    {
        "corpusid": 222,
        "externalids": {"doi": None, "arxiv": "2101.00001", "pubmed": None},
        "content": {
            "source": {"oainfo": {"license": "CCBY-SA", "status": "GREEN",
                                  "openaccessurl": None}},
            "text": "Second body about graphs.",
            "annotations": {},
        },
    },
    {  # no body text -> skipped
        "corpusid": 333,
        "externalids": {"doi": "10.1/notext"},
        "content": {"text": None, "annotations": {}},
    },
    {  # no corpusid -> skipped
        "externalids": {"doi": "10.1/nocid"},
        "content": {"text": "orphan", "annotations": {}},
    },
    {  # Schema B: legacy `body`-based record (s2orc dataset files ~320-635)
        "corpusid": 555,
        "openaccessinfo": {
            "license": "CCBYNCSA", "status": "GOLD",
            "url": "https://doi.org/10.4148/legacy",
            "externalids": {"DOI": "10.4148/legacy", "ArXiv": None,
                            "Medline": "778899", "PubMedCentral": None, "MAG": "3160"},
        },
        "title": "A legacy full-text paper",
        "authors": [{"name": "A. Author"}],
        "body": {"text": "Legacy body full text about ecosystems."},
        "bibliography": [{"title": "ref one"}],
    },
    {  # duplicate corpusid: legacy (no annotations) reduced FIRST ...
        "corpusid": 777,
        "openaccessinfo": {"license": "CCBY", "status": "GREEN", "url": None,
                           "externalids": {}},
        "body": {"text": "Legacy text for a paper also in the modern half."},
    },
    {  # ... then modern (with annotations): modern must win despite order
        "corpusid": 777,
        "externalids": {"doi": "10.9/dup"},
        "content": {"source": {"oainfo": {"license": "CCBY", "status": "GOLD",
                                          "openaccessurl": "https://ex.org/dup.pdf"}},
                    "text": "Modern text for the same paper.",
                    "annotations": {"paragraph": '[{"start":0,"end":6}]'}},
    },
]


def test_prefer_annotated_on_conflict(tmp_path):
    store = _build_store(tmp_path)
    rec = store.get_fulltext(777, want_annos=True)
    # modern record won despite being reduced AFTER the legacy one
    assert rec["text"].startswith("Modern text")
    assert rec["status"] == "GOLD"
    assert "paragraph" in (rec["annotations"] or {})


def test_schema_b_legacy_body(tmp_path):
    b = schema.slim_record(RECORDS[4])
    assert b is not None and b["corpusid"] == 555
    assert b["text"].startswith("Legacy body full text")
    assert b["license"] == "CCBYNCSA" and b["status"] == "GOLD"
    assert b["oaurl"] == "https://doi.org/10.4148/legacy"
    keys = schema.idmap_keys(b["externalids"])
    assert "doi:10.4148/legacy" in keys      # TitleCase DOI matched case-insensitively
    assert "pmid:778899" in keys             # Medline -> pmid
    # round-trips through the store, resolvable by every id kind
    store = _build_store(tmp_path)
    assert store.resolve("CorpusId:555") == 555
    assert store.resolve("DOI:10.4148/legacy") == 555
    assert store.resolve("PMID:778899") == 555
    rec = store.get_fulltext(555)
    assert rec["text"].startswith("Legacy body full text") and rec["license"] == "CCBYNCSA"


def _build_store(tmp_path: Path, records=RECORDS) -> MirrorStore:
    src = tmp_path / "s2orc_000.jsonl.gz"
    with gzip.open(src, "wb") as f:
        for r in records:
            f.write((json.dumps(r) + "\n").encode())
    parts = tmp_path / "parts"
    mapper.map_dataset_file("s2orc", src, parts, 0, scratch=str(tmp_path))
    store_root = tmp_path / "store"
    for kind, (_fn, n) in reducer.REDUCERS.items():
        for shard in range(n):
            reducer.reduce_shard(kind, shard, parts, store_root, scratch=str(tmp_path))
    (store_root / "meta.json").write_text(json.dumps({"release": "test-1"}))
    return MirrorStore(store_root)


# ---- pure helpers ----------------------------------------------------------

def test_slim_record_extracts_and_skips():
    ok = schema.slim_record(RECORDS[0])
    assert ok is not None
    assert ok["corpusid"] == 89669906
    assert ok["text"].startswith("Body one")
    assert ok["license"] == "CCBY" and ok["status"] == "GOLD"
    assert ok["oaurl"] == "https://ex.org/one.pdf"
    assert schema.slim_record(RECORDS[2]) is None   # no text
    assert schema.slim_record(RECORDS[3]) is None   # no corpusid


def test_idmap_keys_lowercase_and_selective():
    keys = schema.idmap_keys(RECORDS[0]["externalids"])
    assert "doi:10.5935/2177-4560.20110018" in keys
    assert "pmid:12345" in keys
    assert not any(k.startswith("mag:") for k in keys)   # mag/acl/dblp not indexed


@pytest.mark.parametrize("raw,expected", [
    ("CorpusId:89669906", ("corpus", 89669906)),
    ("DOI:10.5935/X", ("key", "doi:10.5935/x")),
    ("ARXIV:2101.00001", ("key", "arxiv:2101.00001")),
    ("PMID:12345", ("key", "pmid:12345")),
    ("DOI:10.48550/arXiv.2101.00001", ("key", "arxiv:2101.00001")),  # datacite normalize
    ("MAG:999", None),
    ("", None),
])
def test_parse_paper_id(raw, expected):
    assert schema.parse_paper_id(raw) == expected


# ---- store round-trip ------------------------------------------------------

def test_resolve_by_every_id_kind(tmp_path):
    store = _build_store(tmp_path)
    assert store.resolve("CorpusId:89669906") == 89669906
    assert store.resolve("DOI:10.5935/2177-4560.20110018") == 89669906
    assert store.resolve("PMID:12345") == 89669906
    assert store.resolve("ARXIV:2101.00001") == 222
    assert store.resolve("DOI:10.1/notext") is None   # skipped record, no idmap row


def test_get_fulltext_body_meta_annotations(tmp_path):
    store = _build_store(tmp_path)
    rec = store.get_fulltext(89669906)
    assert rec["text"].startswith("Body one about proteins")
    assert rec["license"] == "CCBY" and rec["status"] == "GOLD"
    assert rec["openAccessUrl"] == "https://ex.org/one.pdf"
    assert rec["externalIds"]["doi"] == "10.5935/2177-4560.20110018"
    assert "annotations" not in rec           # not requested by default
    rec2 = store.get_fulltext(89669906, want_annos=True)
    assert set(rec2["annotations"]) == {"paragraph", "sectionheader"}
    meta_only = store.get_fulltext(89669906, want_text=False)
    assert "text" not in meta_only and meta_only["license"] == "CCBY"


def test_has_and_missing(tmp_path):
    store = _build_store(tmp_path)
    assert store.has(89669906) is True
    assert store.has(333) is False            # no-text record was skipped
    assert store.has(999999) is False
    assert store.get_fulltext(999999) is None


def test_batch_get_and_resolve_many(tmp_path):
    store = _build_store(tmp_path)
    recs = store.get_many_fulltext([89669906, 222, 999999])
    assert set(recs) == {89669906, 222}
    assert recs[222]["text"].startswith("Second body")
    rmap = store.resolve_many(["CorpusId:222", "DOI:10.5935/2177-4560.20110018", "PMID:0"])
    assert rmap["CorpusId:222"] == 222
    assert rmap["DOI:10.5935/2177-4560.20110018"] == 89669906
    assert rmap["PMID:0"] is None


# ---- HTTP surface ----------------------------------------------------------

def test_server_surface(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from s2orc_mirror.server import create_app

    store = _build_store(tmp_path)
    client = TestClient(create_app(store, api_keys={"secret-key"}))

    # auth required
    assert client.get("/s2orc/v1/paper/CorpusId:89669906").status_code == 401
    h = {"x-api-key": "secret-key"}

    r = client.get("/s2orc/v1/paper/CorpusId:89669906", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["text"].startswith("Body one") and body["license"] == "CCBY"

    # resolve by DOI too
    r = client.get("/s2orc/v1/paper/DOI:10.5935/2177-4560.20110018", headers=h)
    assert r.status_code == 200 and r.json()["corpusid"] == 89669906

    # miss -> 404
    assert client.get("/s2orc/v1/paper/CorpusId:999999", headers=h).status_code == 404

    # meta-only omits the body
    r = client.get("/s2orc/v1/paper/CorpusId:89669906?include=meta", headers=h)
    assert "text" not in r.json()

    # batch aligns to input, null for misses
    r = client.post("/s2orc/v1/paper/batch",
                    json={"ids": ["CorpusId:89669906", "CorpusId:999999", "ARXIV:2101.00001"]},
                    headers=h)
    data = r.json()
    assert data[0]["corpusid"] == 89669906
    assert data[1] is None
    assert data[2]["corpusid"] == 222
