"""Sonnet 5.5 vs Fable 5.1 on the LLM arms (a), (b) of the planted-defect control (docs/CONTROLS_REPORT.md §11).

    PYTHONPATH=. .venv/bin/python scripts/aggregate_planted_f1.py

Kraken: for each corruption and arm, the Sonnet run `runs/20261107-planted-<cid>-<arm>-b*-1` and the Fable run
`runs/*-planted-<cid>-<arm>-f1-b*-1`; held-out `blb-eval-hand`, same definitions and classes as scripts/aggregate_planted.py
(recovered share = (CER_corrupted - CER_final) / damage; classes on the page-clustered interval of final vs corrupted).
Adapter p3: score set `kurrent-val-holdout-64`, numbers as in docs/ADAPTER_REPORT.md A3.
Writes docs/controls/planted_f1.json and docs/controls/planted_f1.md.
"""
import json
from pathlib import Path

from wsp.data import load
from wsp.metrics import per_line_edits, bootstrap_delta, page_of
from wsp.plant import CORRUPTIONS, OUT

SC = Path.home() / "atr-cache/wsp/scores"
SET = Path("data/heldout/blb-eval-hand.jsonl")
items = load(SET); refs = [i["text"] for i in items]; PAGES = [page_of(i["key"]) for i in items]
plant = json.loads((OUT / "plant.json").read_text())


def cache(digest):
    f = next(SC.glob(f"heldout-blb-eval-hand-a784ae084058-{digest[:16]}.json"))
    c = json.loads(f.read_text())
    return per_line_edits(c["hyps"], refs), c["metrics"]


orig_lines, orig_m = cache("f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd")
ORDER = [c for c in CORRUPTIONS if c in plant] + [c + "-u" for c in ("k1-zero-row-e", "k6-lstm16-x0.5", "k8-zero-conv-c3")]


def arm_row(run, cl, cm, damage):
    sm = json.loads((run / "summary.json").read_text()); h = sm.get("heldout")
    a = {"run": run.name, "passes": sm["used"], "edits": sm["edits"], "kept": sm["kept"], "final_digest": sm["final_digest"],
         "unchanged": sm["final_digest"] == sm["start_digest"], "scored_from_cache": bool(h and h.get("scored_from_cache"))}
    if not h:
        a["class"] = "no held-out score"
        return a
    fl = [tuple(x) for x in h["per_line"]]
    dd = bootstrap_delta(cl, fl, resamples=2000, seed=20261106, clusters=PAGES)
    fc = h["metrics"]["cer"]
    frac = (cm["cer"] - fc) / damage if abs(damage) > 1e-9 else None
    a.update({"heldout_cer": fc, "delta_vs_corrupted_pts": 100 * dd["delta_cer"], "delta_ci95_pts": [100 * x for x in dd["ci95"]],
              "delta_ci95_pages_pts": [100 * x for x in dd["ci95_cluster"]], "recovered_fraction": frac})
    if damage < 0.005:
        cls = "null corruption"
    elif frac >= 0.8 and dd["ci95_cluster"][1] < 0:
        cls = "repaired"
    elif frac >= 0.2 and dd["ci95_cluster"][1] < 0:
        cls = "partly"
    else:
        cls = "not repaired"
    a["class"] = cls
    return a


out = {"original_heldout_cer": orig_m["cer"], "kraken": {}, "adapter": {}}
for cid in ORDER:
    base = cid[:-2] if cid.endswith("-u") else cid
    cl, cm = cache(plant[base]["digest"])
    damage = cm["cer"] - orig_m["cer"]
    row = {"what": CORRUPTIONS[base][0] + (" [unrelated donor]" if cid.endswith("-u") else ""), "heldout_cer": cm["cer"], "damage_pts": 100 * damage, "arms": {}}
    for arm in "ab":
        for model, pat in (("sonnet", f"20261107-planted-{cid}-{arm}-b*-1"), ("fable", f"*-planted-{cid}-{arm}-f1-b*-1")):
            runs = [r for r in sorted(Path("runs").glob(pat)) if (r / "summary.json").exists()]
            if runs:
                row["arms"][f"{arm}-{model}"] = arm_row(runs[-1], cl, cm, damage)
    out["kraken"][cid] = row

# adapter p3: score set, page metrics (vLLM, not repeatable); clean and corrupted readings from docs/controls/planted-adapter.json
pa = json.loads(Path("docs/controls/planted-adapter.json").read_text()) if Path("docs/controls/planted-adapter.json").exists() else {}
for arm in "ab":
    for model, pat in (("sonnet", f"20261107-planted-adapter-p3-alpha2-{arm}-b*-1"), ("fable", f"*-planted-adapter-p3-alpha2-{arm}-f1-b*-1")):
        runs = [r for r in sorted(Path("runs").glob(pat)) if (r / "summary.json").exists()]
        if not runs:
            continue
        sm = json.loads((runs[-1] / "summary.json").read_text()); h = sm.get("heldout") or {}
        out["adapter"][f"{arm}-{model}"] = {"run": runs[-1].name, "passes": sm["used"], "edits": sm["edits"], "kept": sm["kept"],
                                            "final_digest": sm["final_digest"], "score_set": h.get("metrics")}
out["adapter_reference"] = pa.get("p3-alpha2") or pa
Path("docs/controls/planted_f1.json").write_text(json.dumps(out, indent=1, default=str))

L = ["| corruption | damage (pts) | arm | Sonnet 5.5: held-out CER · recovered · class | passes/edits/kept | Fable 5.1: held-out CER · recovered · class | passes/edits/kept |",
     "|---|---:|---|---|---|---|---|"]
f = lambda x: "—" if x is None else f"{100 * x:.0f} %"
for cid, r in out["kraken"].items():
    for arm in "ab":
        s, fb = r["arms"].get(f"{arm}-sonnet"), r["arms"].get(f"{arm}-fable")
        cell = lambda a: "—" if not a else (f"{100 * a['heldout_cer']:.2f} · {f(a['recovered_fraction'])} · {a['class']}" if "heldout_cer" in a else a["class"])
        pek = lambda a: "—" if not a else f"{a['passes']}/{a['edits']}/{a['kept']}"
        L.append(f"| {cid} | {r['damage_pts']:+.2f} | ({arm}) | {cell(s)} | {pek(s)} | {cell(fb)} | {pek(fb)} |")
Path("docs/controls/planted_f1.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
for k, v in out["adapter"].items():
    print(k, v["run"], v["passes"], v["edits"], v["kept"], {m: (v["score_set"] or {}).get(m) for m in ("cer", "truncated_cer", "at_cap")})
