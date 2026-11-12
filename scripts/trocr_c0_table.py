#!/usr/bin/env python3
"""Aggregate the C0 reports of scripts/trocr_c0.py (~/atr-cache/trocr/c0/{gpu,cpu}/*.json) into a markdown table and docs/trocr/c0_summary.json."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "atr-cache/trocr/c0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp import metrics  # noqa: E402

SETS = {"trocr-blb-train-300": "data/practice/trocr-blb-train-300.jsonl", "blb-train-300": "data/practice/trocr-blb-train-300.jsonl", "blb-eval-hand": "data/heldout/blb-eval-hand.jsonl"}


def spaces(pred_file: Path, set_name: str) -> dict:
    """The BLB references put a space before punctuation (``konnte .``); a model trained on other conventions does not.
    CER with every space removed from both sides, and the share of reference spaces that are missing."""
    refs = {json.loads(l)["key"]: json.loads(l)["text"] for l in open(SETS[set_name], encoding="utf-8")}
    preds = [json.loads(l) for l in open(pred_file, encoding="utf-8")]
    h = [p["text"].replace(" ", "") for p in preds]
    r = [refs[p["key"]].replace(" ", "") for p in preds]
    m = metrics.score(h, r, blocks=False)
    sp_ref = sum(refs[p["key"]].count(" ") for p in preds)
    sp_hyp = sum(p["text"].count(" ") for p in preds)
    return {"cer_without_spaces": m.cer, "spaces_ref": sp_ref, "spaces_hyp": sp_hyp}


rows = []
for p in sorted(root.glob("*/*.json")):
    if p.parent.name not in ("gpu", "cpu", "cpu40"):
        continue
    r = json.loads(p.read_text())
    if "cer" not in r:
        continue
    tf = r.get("teacher_forced") or {}
    rows.append({"model": Path(r["model"]).name.replace("dh-unibe__", ""), "set": Path(r["set"]).stem, "device": r["device"],
                 "lines": r["lines"], "cer": r["cer"], "wer": r["wer"], "collapsed": r["collapsed"], "length_ratio": r["length_ratio"],
                 "tokens": r["tokens_total"], "at_cap": r["at_cap"], "sec": [x["seconds"] for x in r["passes"]],
                 "sec_per_line": [x["sec_per_line"] for x in r["passes"]], "identical": r["byte_identical_repeat"],
                 "started": r["passes"][0]["started"], "peak_gpu_mib": r["passes"][0]["peak_gpu_mib"],
                 "nll": tf.get("mean_token_nll"), "margin_neg": tf.get("share_margin_negative"), "digest": r["digest"][:12],
                 **spaces(p.with_suffix(".pred.jsonl") if p.with_suffix(".pred.jsonl").exists() else Path(str(p)[:-5] + ".pred.jsonl"), Path(r["set"]).stem),
                 "roundtrip": (r.get("roundtrip") or {}).get("equal"), "pred_sha256": r["passes"][0]["pred_sha256"][:12]})
Path("docs/trocr/c0_summary.json").write_text(json.dumps(rows, indent=1))
print("| model | set | device | lines | CER % | WER % | collapsed | length ratio | tokens | at cap | CER no spaces % | s per pass (1st, 2nd) | repeat byte-identical |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
for x in rows:
    print(f"| {x['model']} | {x['set']} | {x['device']} | {x['lines']} | {100 * x['cer']:.2f} | {100 * x['wer']:.2f} | {x['collapsed']} | {x['length_ratio']:.3f} | {x['tokens']} | {x['at_cap']} | {100 * x['cer_without_spaces']:.2f} | {' / '.join(f'{s:.0f}' for s in x['sec'])} | {x['identical']} |")

# cross-device agreement: CPU lines against the same lines on card 0
print()
print("| model | lines compared | texts differing CPU vs card 0 | max abs beam-score difference |")
print("|---|---:|---:|---:|")
for cp in sorted((root / "cpu40").glob("trocr-blb-train-300.*.cpu.pred.jsonl")) if (root / "cpu40").exists() else []:
    m = cp.name[len("trocr-blb-train-300."):-len(".cpu.pred.jsonl")]
    gp = root / "gpu" / f"trocr-blb-train-300.{m}.cuda0.pred.jsonl"
    if not gp.exists():
        continue
    c = [json.loads(l) for l in open(cp, encoding="utf-8")]
    g = [json.loads(l) for l in open(gp, encoding="utf-8")][:len(c)]
    diff = sum(a["text"] != b["text"] for a, b in zip(c, g))
    sc = max(abs((a["extra"]["score"] or 0) - (b["extra"]["score"] or 0)) for a, b in zip(c, g))
    print(f"| {m} | {len(c)} | {diff} | {sc:.1e} |")
