"""Aggregate the planted-defect control on the adapter (A3) into docs/controls/planted-adapter.{json,md}."""
import glob, json
from pathlib import Path
from wsp import plant_adapter as P
from wsp.metrics import bootstrap_delta

REPO = Path(__file__).resolve().parent.parent
B = json.loads((REPO / "docs/baselines/a3-baselines.json").read_text())
NOISE = 5.7   # points of CER: repeat of the clean adapter on 64 pages (0.350 vs 0.407), A0
def sc(a, s):
    return json.loads(next(iter(P.SCORES.glob(f"adapter-kurrent-val-{s}-*-{a}*.json"))).read_text()) if False else None
H = REPO / "data/practice/kurrent-val-holdout-64.jsonl"
clean = P.score(P.CLEAN if hasattr(P, "CLEAN") else Path.home() / "atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v2", H)
rows, out = [], {}
for cid in ("p1-scale3", "p2-zeroB", "p3-alpha2", "p4-randdir", "p5-alpha05", "p6-scale3-late"):
    cor = P.score(P.OUT / cid, H)
    d = (cor["metrics"]["cer"] - clean["metrics"]["cer"]) * 100
    row = {"cid": cid, "corrupted": cor["metrics"], "damage_points": d, "class": "null corruption (below pass noise)" if d < max(NOISE, 3) else "defect", "arms": {}}
    for run in sorted(glob.glob(str(REPO / f"runs/*planted-adapter-{cid}-*/summary.json"))):
        s = json.loads(Path(run).read_text()); arm = run.split(f"{cid}-")[1][0]
        h = s["heldout"]; fin = h["metrics"]
        gain = (cor["metrics"]["cer"] - fin["cer"]) * 100
        bd = bootstrap_delta(cor["per_page"], h["per_page"]) if len(h["per_page"]) == len(cor["per_page"]) else None
        row["arms"][arm] = {"passes": s["used"], "edits": s["edits"], "kept": s["kept"], "cer": fin["cer"], "trunc": fin["truncated_cer"], "at_cap": fin["at_cap"],
                            "recovered_frac": gain / d if d > 0 else None, "delta_vs_corrupted": bd}
    out[cid] = row
(REPO / "docs/controls/planted-adapter.json").write_text(json.dumps({"clean": clean["metrics"], "noise_points": NOISE, "rows": out}, indent=1))
L = ["| defect | corrupted CER / trunc / at cap | damage (pts) | class | " + " | ".join(f"({a}) CER, recovered, kept/edits/passes" for a in "abc") + " |", "|---|---|---:|---|---|---|---|"]
c = clean["metrics"]
L.insert(0, f"Clean adapter on the 64-page score set: CER {c['cer']:.3f}, truncated {c['truncated_cer']:.3f}, {c['at_cap']} at cap. Repeat noise of the clean adapter: {NOISE} points of CER.\n")
for cid, r in out.items():
    m = r["corrupted"]
    cells = []
    for a in "abc":
        x = r["arms"].get(a)
        cells.append("—" if not x else f"{x['cer']:.3f}, " + (f"{x['recovered_frac']*100:.0f} %" if x["recovered_frac"] is not None else "n/a") + f", {x['kept']}/{x['edits']}/{x['passes']}")
    L.append(f"| {cid} | {m['cer']:.3f} / {m['truncated_cer']:.3f} / {m['at_cap']} | {r['damage_points']:+.1f} | {r['class']} | " + " | ".join(cells) + " |")
(REPO / "docs/controls/planted-adapter.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
