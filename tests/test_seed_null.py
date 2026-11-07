"""The seed-null data lists: page-level split, disjoint, clean against the Hertwig test pages."""
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs" / "seed-null"


def _rows(name):
    return [json.loads(l) for l in (D / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def _pages(rows):
    return {re.search(r"/(p\d{4})/", r["key"]).group(1) for r in rows}


def test_split_is_page_level_and_complete():
    tr, va = _rows("train.jsonl"), _rows("val.jsonl")
    assert not _pages(tr) & _pages(va)
    assert len(_pages(tr)) + len(_pages(va)) == 50
    man = json.loads((D / "split.json").read_text())
    assert man["train_lines"] == len(tr) and man["val_lines"] == len(va)
    assert {r["key"] for r in tr}.isdisjoint({r["key"] for r in va})


def test_audit_clean_and_runs_complete():
    audit = json.loads((D / "audit.json").read_text())
    assert all(s["clean"] for s in audit["sets"])
    runs = json.loads((D / "runs.json").read_text())
    assert len(runs) >= 3
    assert all(r["epochs"] == 91 and len(r["val_accuracy_per_epoch"]) == 91 for r in runs.values())
