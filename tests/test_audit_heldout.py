"""wsp.audit: shelfmark spellings and the held-out audit (engine-free)."""
import json

from wsp.audit import audit_heldout, audit_set, norm_id


def _write(tmp_path, name, rows):
    p = tmp_path / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / f"{name}.meta.json").write_text(json.dumps({"sha256": "x", "source": "test"}))
    return p


IDS = {"blb_eval_objects": {"lassberg": ["K2911-14", "K2911-17"]}, "hertwig": {}, "medieval": {}}


def test_norm_id_joins_the_two_spellings():
    assert norm_id("K2911, 14") == norm_id("K2911-14") == norm_id("K2911,14") == "K291114"


def test_practice_audit_catches_eval_object_under_the_comma_spelling(tmp_path):
    p = _write(tmp_path, "prac", [{"key": "a", "doc": "K2911,14", "text": "x"}])
    r = audit_set(p, IDS)
    assert not r["clean"] and r["findings"][0]["kind"] == "blb_eval_object_in_practice"


def test_heldout_audit(tmp_path):
    held = _write(tmp_path, "held", [{"key": "h1", "doc": "K2911,14", "text": "x"}, {"key": "h2", "doc": "K2911,17", "text": "y"}])
    prac_ok = _write(tmp_path, "prac_ok", [{"key": "p", "doc": "K2911,15", "text": "z"}])
    prac_bad = _write(tmp_path, "prac_bad", [{"key": "q", "doc": "K2911-17", "text": "z"}])
    assert audit_heldout(held, [prac_ok], IDS)["clean"]
    r = audit_heldout(held, [prac_ok, prac_bad], IDS)
    assert not r["clean"] and r["findings"][0]["documents"] == ["K2911,17"]
    stray = _write(tmp_path, "stray", [{"key": "s", "doc": "K9999", "text": "x"}])
    assert audit_heldout(stray, [], IDS)["findings"][0]["kind"] == "rows_neither_blb_eval_object_nor_test_project"


def test_heldout_audit_transkribus_pages(tmp_path):
    held = _write(tmp_path, "bern", [{"key": "A_V_1443_Hertwig_test/p0008/l1", "doc": "13922499", "text": "x"}])
    prac = _write(tmp_path, "pr", [{"key": "A_V_1443_Hertwig_train/p0006/l1", "doc": "13922499", "text": "y"}])
    r = audit_heldout(held, [prac], IDS)
    assert r["clean"], r["findings"]                     # same volume, different page: fine
    clash = _write(tmp_path, "clash", [{"key": "A_V_1443_Hertwig_train/p0008/l9", "doc": "13922499", "text": "y"}])
    assert audit_heldout(held, [clash], IDS)["findings"][0]["kind"] == "page_shared_with_practice"
    wrong = _write(tmp_path, "wrong", [{"key": "A_V_1443_Hertwig_train/p0010/l1", "doc": "13922499", "text": "x"}])
    assert audit_heldout(wrong, [], IDS)["findings"][0]["kind"] == "rows_neither_blb_eval_object_nor_test_project"
