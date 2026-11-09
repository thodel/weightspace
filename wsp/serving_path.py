"""Serving-path comparison of a candidate with its unedited model, paired by page (docs/RESULTS.md R4, R12).

    python -m wsp.serving_path <score_report.json> <unedited reading> <candidate reading> --out runs/<run>/serving_path.json

Input is the page list of a `score_ground_truth` report (pages taken by name), as `score_report` returns it. Per page
the CER of both readings; better / worse / equal counts; the median of the page deltas and the character-weighted
delta, each with a paired bootstrap over pages (2,000 resamples, seed 20261106). On R4's pages it gives R4's numbers
(the weighted interval's upper end differs in the fourth decimal of a point: R4's was computed by hand, not kept).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from wsp.metrics import BOOTSTRAP_SEED


def compare(pages: list[dict], unedited: str, candidate: str, resamples: int = 2000, seed: int = BOOTSTRAP_SEED) -> dict:
    rows = []
    for p in pages:
        r = {x["reading"]: x["cer"] for x in p["readings"]}
        rows.append({"file": p["file"], "chars": p["chars"], "cer_unedited": r[unedited], "cer_candidate": r[candidate]})
    a = np.array([r["cer_unedited"] for r in rows]); b = np.array([r["cer_candidate"] for r in rows])
    n = np.array([r["chars"] for r in rows], dtype=float)
    d = b - a
    rng = random.Random(seed)                # as wsp.metrics.bootstrap_delta; reproduces R4's median interval exactly
    med, wt = [], []
    for _ in range(resamples):
        i = [rng.randrange(len(rows)) for _ in rows]
        med.append(np.median(d[i])); wt.append((b[i] * n[i]).sum() / n[i].sum() - (a[i] * n[i]).sum() / n[i].sum())
    med.sort(); wt.sort()
    lo, hi = int(0.025 * resamples), min(resamples - 1, int(0.975 * resamples))
    return {"pages": len(rows), "better": int((d < 0).sum()), "worse": int((d > 0).sum()), "equal": int((d == 0).sum()),
            "median_page_delta_pt": 100 * float(np.median(d)), "median_ci95_pt": [100 * float(med[lo]), 100 * float(med[hi])],
            "weighted_delta_pt": 100 * float((b * n).sum() / n.sum() - (a * n).sum() / n.sum()),
            "weighted_ci95_pt": [100 * float(wt[lo]), 100 * float(wt[hi])],
            "medians": [float(np.median(a)), float(np.median(b))],
            "weighted": [float((a * n).sum() / n.sum()), float((b * n).sum() / n.sum())],
            "resamples": resamples, "seed": seed, "per_page": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.serving_path")
    ap.add_argument("report"); ap.add_argument("unedited"); ap.add_argument("candidate")
    ap.add_argument("--name", default=None, help="report name to record"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = json.loads(Path(a.report).read_text())
    res = compare(rep["pages"], a.unedited, a.candidate)
    res = {"report": a.name or rep.get("name"), **res}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "per_page"})


if __name__ == "__main__":
    main()
