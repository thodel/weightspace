"""E3 (docs/XVENDOR_PLAN.md §2, §7.1): Fable 5.1 vs Gemini 4 Argon vs GPT on the LLM arms (a), (b) of the kraken
planted-defect control.

    PYTHONPATH=. .venv/bin/python scripts/aggregate_planted_xv.py

Same definitions and classes as scripts/aggregate_planted_f1.py (held-out `blb-eval-hand`; recovered share =
(CER_corrupted - CER_final) / damage; classes on the page-clustered interval of final vs corrupted). Two classes added
by §7.1, both counted as not repaired:
  - failed (program error): the program ran but did not freeze (an exception, or no s.freeze());
  - failed (no program): the cell's conversation ended without writing program.py (it is not rerun).
Runs: Fable `runs/*-planted-<cid>-<arm>-f1-b*-1`; cross-vendor `runs/*-planted-<cid>-<arm>-<tag>-b*-1`, cells
`runs/planted-xv/<tag>/<sNN>/<arm>/`. Writes docs/controls/planted_xv.json and docs/controls/planted_xv.md.
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
    """As scripts/aggregate_planted_f1.py arm_row (copied, not imported: that script writes its own tables on import)."""
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


MAP = json.loads(Path("runs/planted-f1/map.json").read_text())
SLOT = {v: k for k, v in MAP.items()}
VENDORS = (("fable", "f1", "Fable 5.1"), ("gemini", "g1", "Gemini 4 Argon"), ("gpt", "o1", "GPT-6.2 Astra"))


def cell(cid, arm, tag, cl, cm, damage):
    if tag != "f1" and not (Path("runs/planted-xv") / tag / SLOT[cid] / arm / "program.py").exists():
        return {"class": "failed (no program)", "repaired": False}
    runs = sorted(Path("runs").glob(f"*-planted-{cid}-{arm}-{tag}-b*-1"))
    if not runs:
        return None
    done = [r for r in runs if (r / "summary.json").exists()]  # summary.json is written by s.freeze()
    if not done:
        return {"run": runs[-1].name, "class": "failed (program error)", "repaired": False}
    a = arm_row(done[-1], cl, cm, damage)
    a["repaired"] = a["class"] == "repaired"
    return a


out = {"original_heldout_cer": orig_m["cer"], "kraken": {}}
for cid in ORDER:
    base = cid[:-2] if cid.endswith("-u") else cid
    cl, cm = cache(plant[base]["digest"])
    damage = cm["cer"] - orig_m["cer"]
    row = {"slot": SLOT[cid], "what": CORRUPTIONS[base][0] + (" [unrelated donor]" if cid.endswith("-u") else ""),
           "heldout_cer": cm["cer"], "damage_pts": 100 * damage, "arms": {}}
    for arm in "ab":
        for key, tag, _ in VENDORS:
            c = cell(cid, arm, tag, cl, cm, damage)
            if c:
                row["arms"][f"{arm}-{key}"] = c
    out["kraken"][cid] = row

CLASSES = ("repaired", "partly", "not repaired", "null corruption", "failed (program error)", "failed (no program)")
tally = {}
for arm in "ab":
    for key, _, _ in VENDORS:
        t = {c: 0 for c in CLASSES}
        for r in out["kraken"].values():
            a = r["arms"].get(f"{arm}-{key}")
            if a:
                t[a["class"]] = t.get(a["class"], 0) + 1
        tally[f"{arm}-{key}"] = t
out["tally"] = tally
Path("docs/controls/planted_xv.json").write_text(json.dumps(out, indent=1, default=str))

f = lambda x: "—" if x is None else f"{100 * x:.0f} %"
show = lambda a: "—" if not a else (f"{100 * a['heldout_cer']:.2f} · {f(a['recovered_fraction'])} · {a['class']}"
                                    if "heldout_cer" in a else a["class"])
L = ["| corruption | damage (pts) | arm | " + " | ".join(f"{n}: held-out CER · recovered · class" for _, _, n in VENDORS) + " |",
     "|---|---:|---|" + "---|" * len(VENDORS)]
for cid, r in out["kraken"].items():
    for arm in "ab":
        L.append(f"| {cid} | {r['damage_pts']:+.2f} | ({arm}) | "
                 + " | ".join(show(r["arms"].get(f"{arm}-{k}")) for k, _, _ in VENDORS) + " |")
L += ["", "| arm | programmer | " + " | ".join(CLASSES) + " |", "|---|---|" + "---:|" * len(CLASSES)]
for arm in "ab":
    for key, _, name in VENDORS:
        L.append(f"| ({arm}) | {name} | " + " | ".join(str(tally[f'{arm}-{key}'][c]) for c in CLASSES) + " |")
Path("docs/controls/planted_xv.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
