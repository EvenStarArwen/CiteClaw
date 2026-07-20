"""``build_rejected_page`` — the Run sidebar's Rejected-tab pagination.

Reads ``ctx.rejection_details`` (bounded display dicts), hides papers that
ultimately landed in ``ctx.collection``, and serves newest-first pages with
tunable sort/limit/offset.
"""

from __future__ import annotations

from types import SimpleNamespace

from web.live.backend.snapshots import build_rejected_page


def _detail(pid, **kw):
    d = {"id": pid, "title": f"T{pid}", "authors": "", "year": 2020,
         "venue": "", "cites": 0, "depth": 1, "source": "forward",
         "reason": "r", "category": "year"}
    d.update(kw)
    return d


def _ctx(details, collection=None, rejected=None):
    return SimpleNamespace(
        rejection_details=dict(details),
        collection=dict(collection or {}),
        rejected=set(rejected if rejected is not None else details.keys()),
    )


def test_recent_is_newest_first_and_paginates():
    details = {f"p{i}": _detail(f"p{i}") for i in range(5)}   # insertion order p0..p4
    ctx = _ctx(details)
    page = build_rejected_page(ctx, offset=0, limit=2, sort="recent")
    assert [x["id"] for x in page["items"]] == ["p4", "p3"]   # newest first
    assert page["total"] == 5 and page["offset"] == 0 and page["limit"] == 2
    page2 = build_rejected_page(ctx, offset=2, limit=2, sort="recent")
    assert [x["id"] for x in page2["items"]] == ["p2", "p1"]


def test_accepted_papers_are_hidden():
    """A paper rejected by one branch but ultimately accepted (in
    collection) must not appear in the Rejected tab."""
    details = {"p0": _detail("p0"), "p1": _detail("p1")}
    ctx = _ctx(details, collection={"p1": object()})
    page = build_rejected_page(ctx)
    assert [x["id"] for x in page["items"]] == ["p0"]
    assert page["total"] == 1


def test_sort_by_cites_then_year():
    details = {
        "a": _detail("a", cites=10, year=2019),
        "b": _detail("b", cites=50, year=2024),
        "c": _detail("c", cites=30, year=2021),
    }
    ctx = _ctx(details)
    assert [x["id"] for x in build_rejected_page(ctx, sort="cites")["items"]] == ["b", "c", "a"]
    assert [x["id"] for x in build_rejected_page(ctx, sort="year")["items"]] == ["b", "c", "a"]


def test_sort_by_category_groups():
    details = {
        "a": _detail("a", category="year"),
        "b": _detail("b", category="citation"),
        "c": _detail("c", category="year"),
    }
    page = build_rejected_page(_ctx(details), sort="category")
    cats = [x["category"] for x in page["items"]]
    assert cats == sorted(cats)


def test_limit_and_offset_clamped():
    ctx = _ctx({f"p{i}": _detail(f"p{i}") for i in range(3)})
    page = build_rejected_page(ctx, offset=-5, limit=0, sort="recent")
    assert page["offset"] == 0 and page["limit"] >= 1
    # limit is capped so a hostile ?limit=99999 can't dump everything at once
    big = build_rejected_page(ctx, limit=10_000)
    assert big["limit"] <= 200


def test_unknown_sort_falls_back_to_recent():
    ctx = _ctx({"a": _detail("a"), "b": _detail("b")})
    page = build_rejected_page(ctx, sort="bogus")
    assert page["sort"] == "recent"
    assert [x["id"] for x in page["items"]] == ["b", "a"]


def test_capped_flag(monkeypatch):
    import citeclaw.filters.runner as runner_mod
    monkeypatch.setattr(runner_mod, "MAX_REJECTION_DETAILS", 2)
    assert build_rejected_page(_ctx({"p0": _detail("p0"), "p1": _detail("p1")}))["capped"] is True
    assert build_rejected_page(_ctx({"p0": _detail("p0")}))["capped"] is False


def test_empty_context():
    page = build_rejected_page(_ctx({}))
    assert page["total"] == 0 and page["items"] == []
