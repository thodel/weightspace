#!/usr/bin/env python3
"""Seed-42 repeat check (docs/TROCR_NULLS_PREREG.md): CER on the job's 500 eval lines and distance to from-kurrent-v2.

    python scripts/trocr_seed_check.py ~/atr-cache/trocr/seed-null/s42 --out docs/trocr/seed42_check.json [--device cuda:0]

The 500 lines are those of the job's eval_report.raw.jsonl (read only). The recorded predictions are re-scored with wsp.metrics
(NFD CER) so that the comparison does not depend on the job's own scorer.
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
JOB = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs/20261026T064540Z-trocr-german-kurrent-xvi-xix-from-kurrent-v2")
CACHE = Path.home() / "atr-cache" / "trocr"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--skip-ref", action="store_true", help="do not re-run from-kurrent-v2 on the 500 lines")
a = ap.parse_args()
if a.device.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from wsp import metrics
from wsp.trocr_subject import TrocrSubject
from wsp import trocr_family as fam

rows = [json.loads(l) for l in (JOB / "data/eval_report.raw.jsonl").read_text().splitlines()]
items = [{"key": r["image"], "text": r["reference"], "image": Image.open(JOB / r["image"]).convert("RGB")} for r in rows]
refs = [r["reference"] for r in rows]
out = {"lines": len(rows), "recorded_job_cer_own_scorer": metrics.score([r["prediction"] for r in rows], refs, blocks=False).cer}


def run(path):
    sub = TrocrSubject(path, device=a.device, batch=8)
    p = sub.forward(items)
    return metrics.score([x.text for x in p], refs, blocks=False).cer


out["model_cer"] = run(a.model)
if not a.skip_ref:
    out["from_kurrent_v2_cer"] = run(str(CACHE / fam.MODELS["from-kurrent-v2"]))
# distance, stage-1 definition (all tensors the forward uses; weights only), against from-kurrent-v2 and the parent
tmp = Path(a.out).with_suffix(".readings.json")
res = fam.read(tmp, {"cand": str(Path(a.model).relative_to(CACHE)) if str(a.model).startswith(str(CACHE)) else a.model,
                     "from-kurrent-v2": fam.MODELS["from-kurrent-v2"], "xvi-xvii": fam.MODELS["xvi-xvii"]})
sel_all = lambda r, m: not m["unused"] and m["component"] != "other"
sel_w = lambda r, m: sel_all(r, m) and m["kind"] == "weight"
Da, Dw = fam.dist_matrix(res, sel_all), fam.dist_matrix(res, sel_w)
out["dist_all"] = {"cand_to_fkv2": Da[("cand", "from-kurrent-v2")], "cand_to_parent": Da[("cand", "xvi-xvii")], "fkv2_to_parent": Da[("from-kurrent-v2", "xvi-xvii")]}
out["dist_weights"] = {"cand_to_fkv2": Dw[("cand", "from-kurrent-v2")], "cand_to_parent": Dw[("cand", "xvi-xvii")], "fkv2_to_parent": Dw[("from-kurrent-v2", "xvi-xvii")]}
tmp.unlink()
out["limits"] = {"cer_abs_diff_from_recorded_8.89": 0.006, "dist_all_max": 0.044}
out["pass"] = abs(out["model_cer"] - out["recorded_job_cer_own_scorer"]) <= 0.006 and out["dist_all"]["cand_to_fkv2"] <= 0.044
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
