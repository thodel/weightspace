"""Recompute every interval of docs/RESULTS.md and the phase reports with the page-clustered bootstrap (brief item 4).

    .venv/bin/python scripts/recompute_intervals.py   ->  docs/controls/cluster_intervals.{md,json}

Line-level numbers are reproduced exactly (same seed) and sit beside the cluster interval (pages resampled, then lines within
a page; 2,000 resamples, seed 20261106). Sources: the committed per-line baselines, the sessions' `summary.json` and the
sessions' pass readings in the main checkout (git-ignored `runs/*/readings`).
"""
import json
from pathlib import Path

from wsp.data import load
from wsp.metrics import per_line_edits, bootstrap_delta, page_of

MAIN = Path.home() / "Repo/weightspace/runs"
rows = []


def add(label, before, after, keys, source):
    d = bootstrap_delta(before, after, 2000, 20261106, clusters=[page_of(k) for k in keys])
    rows.append({"label": label, "lines": d["lines"], "pages": d["clusters"], "delta_pts": 100 * d["delta_cer"],
                 "line_ci": [100 * x for x in d["ci95"]], "page_ci": [100 * x for x in d["ci95_cluster"]], "source": source})


def baseline_lines(path, model):
    return [tuple(x) for x in next(m for m in json.load(open(path))["models"] if m["model"] == model)["per_line"]]


def kept_pass(run):
    ev = [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]
    last = None; cur = 0
    for e in ev:
        if e["kind"] == "pass": cur += 1
        if e["kind"] == "decision" and e["kept"]: last = cur
    return last


def practice_after(run, prac):
    items = load(prac); refs = {i["key"]: i["text"] for i in items}
    n = kept_pass(run)
    rd = [json.loads(l) for l in open(MAIN / run / "readings" / f"pass_{n:03d}.jsonl")]
    return per_line_edits([r["text"] for r in rd], [refs[r["key"]] for r in rd]), n


def heldout(run):
    sm = json.load(open(f"runs/{run}/summary.json"))["heldout"]
    return [tuple(x) for x in sm["per_line"]]


# R2: session 1 (xix-blb-v1), practice blb-train-300, held-out blb-eval, re-aggregated on blb-eval-hand
R = "20261106-german-xix-blb-v1-b10-1"
pb = baseline_lines("docs/baselines/blb-train-300.json", "kraken-german-xix-blb-v1")
pa, n = practice_after(R, "data/practice/blb-train-300.jsonl")
add(f"R2 session 1, practice (pass {n} vs C0)", pb, pa, [i["key"] for i in load("data/practice/blb-train-300.jsonl")], "PHASE1 §2")
hb = baseline_lines("docs/baselines/heldout-blb-eval.json", "kraken-german-xix-blb-v1"); ha = heldout(R)
hk = [i["key"] for i in load("data/heldout/blb-eval.jsonl")]
add("R2 session 1, held-out blb-eval (5,628 lines)", hb, ha, hk, "PHASE1 §3, RESULTS R2")
for proj in ("lassberg_eval", "korrespondenten_eval", "unbestimmt_eval", "manuskripte_eval"):
    idx = [i for i, k in enumerate(hk) if k.startswith(proj + "/")]
    add(f"R2 session 1, held-out {proj}", [hb[i] for i in idx], [ha[i] for i in idx], [hk[i] for i in idx], "PHASE1 §3")
hand = {i["key"] for i in load("data/heldout/blb-eval-hand.jsonl")}
idx = [i for i, k in enumerate(hk) if k in hand]
add("R2 session 1, re-aggregated on blb-eval-hand (2,272 lines)", [hb[i] for i in idx], [ha[i] for i in idx], [hk[i] for i in idx], "RESULTS R2 note, PHASE5 §2")

# R3: session 2 (combined-v1), practice bern-train-300, held-out bern-eval
R = "20261106-bern-combined-v1-b10-1"
pb = baseline_lines("docs/baselines/bern-train-300.json", "kraken-bern-combined-v1")
pa, n = practice_after(R, "data/practice/bern-train-300.jsonl")
add(f"R3 session 2, practice (pass {n} vs C0)", pb, pa, [i["key"] for i in load("data/practice/bern-train-300.jsonl")], "PHASE3 §3")
hb = baseline_lines("docs/baselines/heldout-bern-eval.json", "kraken-bern-combined-v1"); ha = heldout(R)
hk = [i["key"] for i in load("data/heldout/bern-eval.jsonl")]
add("R3 session 2, held-out bern-eval, all (545)", hb, ha, hk, "PHASE3 §3, RESULTS R3")
for nm in ("Hertwig", "Schlussredaktion"):
    idx = [i for i, k in enumerate(hk) if nm in k]
    add(f"R3 session 2, held-out {nm}", [hb[i] for i in idx], [ha[i] for i in idx], [hk[i] for i in idx], "PHASE3 §3")

# this report: validation sessions, combined-v1 a1 (= a2 = a3) and replay, vs the cached unedited reading
import glob
SC = Path.home() / "atr-cache/wsp/scores"
c = json.loads(next(SC.glob("heldout-bern-eval-*-4836a9f494a9a62c.json")).read_text())
items = load("data/heldout/bern-eval.jsonl"); before = per_line_edits(c["hyps"], [i["text"] for i in items])
for att in ("a1", "replay"):
    r = sorted(glob.glob(f"runs/*val-bern-combined-v1-{att}-*"))[-1]
    fl = [tuple(x) for x in json.load(open(f"{r}/summary.json"))["heldout"]["per_line"]]
    add(f"CONTROLS §3.3 combined-v1 {att}, held-out bern-eval", before, fl, hk, "CONTROLS_REPORT §3.3")

Path("docs/controls").mkdir(exist_ok=True)
Path("docs/controls/cluster_intervals.json").write_text(json.dumps(rows, indent=1))
L = ["| result | lines | pages | Δ (points) | 95 % line-level | 95 % page-clustered | contains 0 (lines → pages) |", "|---|---:|---:|---:|---|---|---|"]
for r in rows:
    zl = r["line_ci"][0] <= 0 <= r["line_ci"][1]; zp = r["page_ci"][0] <= 0 <= r["page_ci"][1]
    L.append(f"| {r['label']} | {r['lines']} | {r['pages']} | {r['delta_pts']:+.3f} | [{r['line_ci'][0]:+.3f}, {r['line_ci'][1]:+.3f}] | "
             f"[{r['page_ci'][0]:+.3f}, {r['page_ci'][1]:+.3f}] | {'yes' if zl else 'no'} → {'yes' if zp else 'no'} |")
Path("docs/controls/cluster_intervals.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
