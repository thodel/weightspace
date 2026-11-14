"""The C4 control on the uncorrected pairs: 44 Hertwig pages from the job directory, text equal to the corrected pages, six runs of five epochs."""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs" / "c4u"


def test_split_and_runs():
    man = json.loads((D / "split.json").read_text())
    assert man["hertwig_train_pages"] == 44 and man["text_identical"] == 44 and man["image_differs"] == 40
    assert sum(p["lines"] for p in man["pages"]) == 1772
    runs = json.loads((D / "runs.json").read_text())
    assert sorted(runs) == [f"U{a}_seed{i}" for a in "AB" for i in (1, 2, 3)]
    assert all(r["epochs"] == 5 for r in runs.values())
