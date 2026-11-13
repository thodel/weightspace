import json
import time

from wsp import card


def _unreachable(*a, **k):
    raise OSError("network unreachable")


def test_waiting_falls_back_to_a_fresh_watcher_file(tmp_path, monkeypatch):
    monkeypatch.setattr(card, "_ask", _unreachable)
    monkeypatch.setattr(card, "STATE", tmp_path / "s.json")
    (tmp_path / "s.json").write_text(json.dumps({"time": time.time(), "waiting": None}))
    assert card.waiting() is None


def test_stale_or_missing_watcher_file_refuses_the_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(card, "_ask", _unreachable)
    monkeypatch.setattr(card, "STATE", tmp_path / "s.json")
    monkeypatch.setattr(card, "free_mib", lambda c=0: 40000)
    ok, why = card.ready()
    assert not ok and "no watcher state" in why
    (tmp_path / "s.json").write_text(json.dumps({"time": time.time() - 600, "waiting": None}))
    ok, why = card.ready()
    assert not ok and "s old" in why
