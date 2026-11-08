"""Tables for docs/CONTROLS_REPORT.md §2 from runs/*planted*: per corruption and arm, held-out numbers with paired intervals."""
import json
from pathlib import Path

from wsp.data import load
from wsp.metrics import score, per_line_edits, bootstrap_delta, page_of
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
out = {"original": {"heldout_cer": orig_m["cer"]}, "corruptions": {}}
ORDER = [c for c in CORRUPTIONS if c in plant] + [c + "-u" for c in ("k1-zero-row-e", "k6-lstm16-x0.5", "k8-zero-conv-c3")]
for cid in ORDER:
    base = cid[:-2] if cid.endswith("-u") else cid
    cl, cm = cache(plant[base]["digest"])
    d = bootstrap_delta(orig_lines, cl, resamples=2000, seed=20261106, clusters=PAGES)
    damage = cm["cer"] - orig_m["cer"]
    prac = json.loads((SC / f"prac-{base}.json").read_text())["metrics"]
    row = {"what": CORRUPTIONS[base][0] + (" [unrelated donor]" if cid.endswith("-u") else ""), "practice_cer": prac["cer"], "heldout_cer": cm["cer"], "damage_pts": 100 * damage,
           "damage_ci95_pts": [100 * x for x in d["ci95"]], "damage_ci95_pages_pts": [100 * x for x in d["ci95_cluster"]], "arms": {}}
    for arm in "abcd":
        runs = sorted(Path("runs").glob(f"*planted-{cid}-{arm}-*"))
        if not runs:
            continue
        r = runs[-1]; sm = json.loads((r / "summary.json").read_text()); h = sm.get("heldout")
        a = {"run": r.name, "passes": sm["used"], "edits": sm["edits"], "kept": sm["kept"], "practice_before": prac["cer"],
             "practice_last_pass": sm["last_pass"]["cer"] if sm["last_pass"] else None, "final_digest": sm["final_digest"]}
        if h:
            fl = h["per_line"]; dd = bootstrap_delta(cl, [tuple(x) for x in fl], resamples=2000, seed=20261106, clusters=PAGES)
            do = bootstrap_delta(orig_lines, [tuple(x) for x in fl], resamples=2000, seed=20261106)
            fc = h["metrics"]["cer"]
            frac = (cm["cer"] - fc) / damage if abs(damage) > 1e-9 else None
            a.update({"heldout_cer": fc, "delta_vs_corrupted_pts": 100 * dd["delta_cer"], "delta_ci95_pts": [100 * x for x in dd["ci95"]], "delta_ci95_pages_pts": [100 * x for x in dd["ci95_cluster"]],
                      "vs_original_pts": 100 * do["delta_cer"], "vs_original_ci95_pts": [100 * x for x in do["ci95"]],
                      "recovered_fraction": frac, "unchanged": sm["final_digest"] == sm["start_digest"]})
            if damage < 0.005:
                cls = "null corruption"
            elif frac >= 0.8 and dd["ci95_cluster"][1] < 0:
                cls = "repaired"
            elif frac >= 0.2 and dd["ci95_cluster"][1] < 0:
                cls = "partly"
            else:
                cls = "not repaired"
            a["class"] = cls
        row["arms"][arm] = a
    out["corruptions"][cid] = row
Path("docs/controls").mkdir(parents=True, exist_ok=True)
Path("docs/controls/planted.json").write_text(json.dumps(out, indent=1))
print("| corruption | prac CER | held-out CER | damage (pts) | " + " | ".join(f"({a}) held-out CER · Δ vs corrupted [CI] · class" for a in "abcd") + " |")
for cid, r in out["corruptions"].items():
    cells = []
    for a in "abcd":
        x = r["arms"].get(a)
        cells.append("—" if not x or "heldout_cer" not in x else
                     f"{100*x['heldout_cer']:.2f} · {x['delta_vs_corrupted_pts']:+.2f} [{x['delta_ci95_pages_pts'][0]:+.2f}, {x['delta_ci95_pages_pts'][1]:+.2f}] (lines [{x['delta_ci95_pts'][0]:+.2f}, {x['delta_ci95_pts'][1]:+.2f}]) · {x['class']}")
    print(f"| {cid} | {100*r['practice_cer']:.2f} | {100*r['heldout_cer']:.2f} | {r['damage_pts']:+.2f} | " + " | ".join(cells) + " |")
