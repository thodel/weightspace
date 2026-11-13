"""The C4 data lists: page-level split of the combined-v1 job, audited clean; six runs with five epochs."""
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs" / "c4"


def _rows(name):
    return [json.loads(l) for l in (D / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def test_split_counts_and_disjoint():
    man = json.loads((D / "split.json").read_text())
    assert man["page_counts"] == {"train_hertwig": 44, "val_hertwig": 6, "train_schluss": 107, "val_schluss": 11}
    tr, va = _rows("train.jsonl"), _rows("val.jsonl")
    assert len(tr) == man["train_lines"] and len(va) == man["val_lines"]
    assert tr and va and {r["key"] for r in tr}.isdisjoint({r["key"] for r in va})
    assert len(tr) + len(va) == 5441            # the trainer's lines_written for the job


def test_audit_clean_and_runs_complete():
    assert all(s["clean"] for s in json.loads((D / "audit.json").read_text())["sets"])
    assert json.loads((D / "audit-validation.json").read_text())["clean"]
    runs = json.loads((D / "runs.json").read_text())
    assert sorted(runs) == [f"{a}_seed{i}" for a in "AB" for i in (1, 2, 3)]
    assert all(r["epochs"] == 5 for r in runs.values())
