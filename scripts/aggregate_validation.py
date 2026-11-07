"""Tables for docs/CONTROLS_REPORT.md §3 from runs/*-val-*."""
import json, glob
from pathlib import Path
from wsp.data import load
from wsp.metrics import per_line_edits, bootstrap_delta

SC = Path.home() / "atr-cache/wsp/scores"
CFG = {"xix-blb-v1": ("blb-eval-hand-b", "f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd"),
       "bern-combined-v1": ("bern-eval", "4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c")}
out = {}
for model, (hs, dig) in CFG.items():
    items = load(f"data/heldout/{hs}.jsonl"); refs = [i["text"] for i in items]
    c = json.loads(next(SC.glob(f"heldout-{hs}-*-{dig[:16]}.json")).read_text())
    before = per_line_edits(c["hyps"], refs); bcer = c["metrics"]["cer"]
    for att in ("a1", "a2", "a3", "replay"):
        r = sorted(Path("runs").glob(f"*val-{model}-{att}-*"))[-1]
        sm = json.loads((r / "summary.json").read_text()); ev = [json.loads(l) for l in open(r / "log.jsonl")]
        kept, h, cur = [], None, None
        for e in ev:
            if e["kind"] == "hypothesis": h = e["text"][:90]
            if e["kind"] == "edit": cur = h
            if e["kind"] == "decision" and e["kept"]: kept.append(cur)
        passes = [e for e in ev if e["kind"] == "pass"]
        refused = sum(1 for e in ev if e["kind"] == "decision" and not e["kept"] and e.get("rule") == "refused")
        reverted = sum(1 for e in ev if e["kind"] == "decision" and not e["kept"] and e.get("rule") != "refused")
        hd = sm["heldout"]; fl = [tuple(x) for x in hd["per_line"]]
        d = bootstrap_delta(before, fl, resamples=2000, seed=20261106)
        out.setdefault(model, {})[att] = {"run": r.name, "passes": sm["used"], "edits": sm["edits"], "kept": sm["kept"], "refused_by_rule": refused,
            "reverted_by_programmer": reverted, "kept_edits": kept, "practice_c0": sm["baseline"]["cer"], "practice_last_pass": sm["last_pass"]["cer"],
            "practice_vs_baseline_ci": passes[-1]["vs_baseline"]["ci95"] if passes else None, "heldout_before": bcer, "heldout_after": hd["metrics"]["cer"],
            "delta_pts": 100 * d["delta_cer"], "ci95_pts": [100 * x for x in d["ci95"]], "unchanged": sm["final_digest"] == sm["start_digest"]}
Path("docs/controls").mkdir(parents=True, exist_ok=True)
Path("docs/controls/validation.json").write_text(json.dumps(out, indent=1))
for m, d in out.items():
    print(m)
    for a, x in d.items():
        print(f"| {a} | {x['passes']} | {x['edits']} | {x['kept']} | {100*x['practice_c0']:.2f} → {100*x['practice_last_pass']:.2f} | {100*x['heldout_before']:.2f} → {100*x['heldout_after']:.2f} | {x['delta_pts']:+.3f} [{x['ci95_pts'][0]:+.3f}, {x['ci95_pts'][1]:+.3f}] | {'; '.join(x['kept_edits']) or '—'} |")
