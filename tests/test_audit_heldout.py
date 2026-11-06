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
    assert audit_heldout(stray, [], IDS)["findings"][0]["kind"] == "document_not_a_blb_eval_object"
