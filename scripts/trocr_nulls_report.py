#!/usr/bin/env python3
"""Stage 2b report: the seed null against the documented traces of stage 1 (docs/TROCR_NULLS_PREREG.md, with its amendment).

    python scripts/trocr_nulls_report.py            # reads ~/atr-cache/trocr/seed-null/s{1..5,42}, docs/trocr/readings.json, displacement.json
                                                    # writes docs/trocr/nulls_readings.json, nulls_summary.json, nulls_tables.md

Samples: ``s1``..``s5`` are the different-seed null (parent ``xvi-xvii``, same data and recipe as ``from-kurrent-v2``); ``s42`` is the same-seed
repeat; ``from-kurrent-v2`` is the recorded run. The survive criterion is the preregistered one: a documented value survives when it lies outside
the range of the different-seed samples AND at least two seed standard deviations from their mean; inside the range it is "not distinguishable";
otherwise "borderline". Seed standard deviations use n-1 over five samples. Run with the TrOCR venv (numpy, safetensors).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp import trocr_family as fam  # noqa: E402
from wsp.trocr_subject import UNUSED, canonical_name  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CACHE = fam.CACHE
SEEDS = ["s1", "s2", "s3", "s4", "s5"]
SAME = "s42"
REC = "from-kurrent-v2"
PARENT = "xvi-xvii"
MODELS = {PARENT: fam.MODELS[PARENT], REC: fam.MODELS[REC], SAME: "seed-null/s42", **{s: f"seed-null/{s}" for s in SEEDS}}
OUTR = REPO / "docs/trocr/nulls_readings.json"

if OUTR.exists() and "--reread" not in sys.argv:
    null = json.loads(OUTR.read_text())
else:
    null = fam.read(OUTR, MODELS)
core = json.loads((REPO / "docs/trocr/readings.json").read_text())
disp = json.loads((REPO / "docs/trocr/displacement.json").read_text())


# ------------------------------------------------------------------------------------------------ metrics of one step (child <- parent)
def _D(res, sel):
    return fam.dist_matrix(res, lambda r, m: (not m["unused"]) and m["component"] != "other" and sel(m))


def step_metrics(res, child, parent):
    meta = res["tensors"]
    enc = _D(res, lambda m: m["component"].startswith("enc."))[(child, parent)]
    dec = _D(res, lambda m: m["component"].startswith("dec."))[(child, parent)]
    dne = _D(res, lambda m: m["component"].startswith("dec.") and m["component"] != "dec.embed")[(child, parent)]
    emb = _D(res, lambda m: m["component"] == "dec.embed")[(child, parent)]
    allw = _D(res, lambda m: True)[(child, parent)]
    # share of the squared move that lies in the tied embedding
    sq = {"emb": 0.0, "all": 0.0}
    for r in res["pairs"]:
        if {r["a"], r["b"]} == {child, parent}:
            m = meta[r["t"]]
            if m["unused"] or m["component"] == "other":
                continue
            sq["all"] += r["sq"]
            if m["component"] == "dec.embed":
                sq["emb"] += r["sq"]
    layers = {}
    for side in ("enc", "dec"):
        v = []
        for i in range(12):
            D = _D(res, lambda m, i=i, side=side: m["kind"] == "weight" and m["layer"] == f"{i:02d}" and m["component"].startswith(side))
            v.append(D[(child, parent)])
        layers[side] = v
    e = layers["enc"]
    hump = float(np.mean(e[3:7]) / np.mean(e[10:12]))
    tr = meta["dec.embed.tokens"]["token_rows"]
    t = tr.get(f"{child}|{parent}") or tr.get(f"{parent}|{child}")
    dec_rows = t["rel_decile_mean"]
    return {"enc": enc, "dec": dec, "dec_no_emb": dne, "emb": emb, "all": allw, "enc_over_dec": enc / dec,
            "emb_share_sq": sq["emb"] / sq["all"], "enc_hump": hump, "enc_peak_layer": int(np.argmax(e)),
            "rows_d9_over_d0": dec_rows[9] / dec_rows[0], "rows_d0": dec_rows[0], "rows_d9": dec_rows[9], "enc_layers": e, "dec_layers": layers["dec"]}


nm = {c: step_metrics(null, c, PARENT) for c in [REC, SAME] + SEEDS}
doc = {f"{c}<-{p}": step_metrics(core, c, p) for c, p in core["parent"].items()}

# ------------------------------------------------------------------------------------------------ move directions (cosines), from the tensors
GROUPS = {"encoder": lambda cn: cn.startswith("enc."), "decoder without embedding": lambda cn: cn.startswith("dec.") and not cn.startswith("dec.embed"),
          "decoder embedding (tied)": lambda cn: cn.startswith("dec.embed")}
kids = [REC, SAME] + SEEDS
pairs = list(itertools.combinations(kids, 2))
handles = {k: __import__("safetensors").safe_open(str(CACHE / v / "model.safetensors"), framework="np") for k, v in MODELS.items()}
names = {canonical_name(k): k for k in handles[PARENT].keys()}
acc = {g: {p: 0.0 for p in pairs} for g in GROUPS}
nrm = {g: {k: 0.0 for k in kids} for g in GROUPS}
for cn, hf in names.items():
    g = next((g for g, f in GROUPS.items() if f(cn)), None)
    if g is None or cn.split(".")[0] + "." + cn.split(".")[1] in UNUSED:
        continue
    p0 = handles[PARENT].get_tensor(hf).astype(np.float32)
    mv = {k: (handles[k].get_tensor(hf).astype(np.float32) - p0).ravel() for k in kids}
    for k in kids:
        nrm[g][k] += float(np.dot(mv[k], mv[k]))
    for a, b in pairs:
        acc[g][(a, b)] += float(np.dot(mv[a], mv[b]))
cos = {g: {f"{a}|{b}": acc[g][(a, b)] / np.sqrt(nrm[g][a] * nrm[g][b]) for a, b in pairs} for g in GROUPS}


# ------------------------------------------------------------------------------------------------ verdicts
def verdict(v, xs):
    xs = np.asarray(xs, float)
    lo, hi, mu, sd = xs.min(), xs.max(), xs.mean(), xs.std(ddof=1)
    if lo <= v <= hi:
        return "not distinguishable (inside the seed range)"
    if abs(v - mu) >= 2 * sd:
        return "survives (outside the range, >= 2 sd)"
    return "borderline (outside the range, < 2 sd)"


out = []
P = out.append
fmt = lambda x: f"{x:.3f}"
P("### N1. The seed null: one step of the recorded recipe (parent `xvi-xvii`, 3 epochs, same data), child <- parent\n")
P("`from-kurrent-v2` is the recorded run (seed 42); `s42` repeats it (same-seed); `s1`..`s5` differ in seed only. Pooled relative L2 as in stage 1 (T4).\n")
keys = [("enc", "encoder, all"), ("dec", "decoder, all"), ("dec_no_emb", "decoder without embedding"), ("emb", "decoder embedding (tied)"),
        ("all", "all tensors"), ("enc_over_dec", "encoder / decoder"), ("emb_share_sq", "share of the squared move in the tied embedding"),
        ("enc_hump", "encoder hump H = mean(layers 3-6) / mean(layers 10-11)"), ("rows_d9_over_d0", "token rows: decile 9 / decile 0 of id")]
P("| reading | recorded | s42 (same seed) | s1 | s2 | s3 | s4 | s5 | seed mean | seed sd | seed range |")
P("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
summ = {}
for k, lab in keys:
    xs = [nm[s][k] for s in SEEDS]
    summ[k] = {"recorded": nm[REC][k], "same_seed": nm[SAME][k], "seeds": xs, "mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1))}
    P(f"| {lab} | {fmt(nm[REC][k])} | {fmt(nm[SAME][k])} | " + " | ".join(fmt(x) for x in xs) + f" | {fmt(np.mean(xs))} | {fmt(np.std(xs, ddof=1))} | {fmt(min(xs))}..{fmt(max(xs))} |")

P("\n### N2. Distance between runs (all tensors the forward uses; weight matrices only in the second block)\n")
for lab, sel in (("all tensors", lambda m: True), ("weight matrices", lambda m: m["kind"] == "weight")):
    D = _D(null, sel)
    P(f"\n{lab}:\n")
    P("| | " + " | ".join(f"`{k}`" for k in [PARENT] + kids) + " |")
    P("|---|" + "---:|" * (len(kids) + 1))
    for a in [PARENT] + kids:
        P(f"| `{a}` | " + " | ".join("—" if a == b else fmt(D[(a, b)]) for b in [PARENT] + kids) + " |")
    ss = [D[(a, b)] for a, b in itertools.combinations(SEEDS, 2)]
    P(f"\nDifferent-seed pairs (10): mean {np.mean(ss):.3f}, range {min(ss):.3f}..{max(ss):.3f}; same-seed pair `{SAME}`-`{REC}`: {D[(SAME, REC)]:.3f}; child-to-parent of the seeds: mean {np.mean([D[(s, PARENT)] for s in SEEDS]):.3f}.")
    summ[f"pair_dist_{'all' if lab[0] == 'a' else 'weights'}"] = {"seed_pairs": ss, "same_seed_pair": D[(SAME, REC)]}

P("\n### N3. Direction of the moves: cosine between moves parent -> child\n")
P("| group | same-seed `s42`~`from-kurrent-v2` | `from-kurrent-v2` ~ s1..s5 (mean, range) | s42 ~ s1..s5 (mean) | seed pairs s_i~s_j (10; mean, range) |")
P("|---|---:|---|---:|---|")
for g in GROUPS:
    c = cos[g]
    get = lambda a, b: c.get(f"{a}|{b}", c.get(f"{b}|{a}"))
    r1 = [get(REC, s) for s in SEEDS]; r2 = [get(SAME, s) for s in SEEDS]; r3 = [get(a, b) for a, b in itertools.combinations(SEEDS, 2)]
    P(f"| {g} | {get(REC, SAME):.3f} | {np.mean(r1):.3f} ({min(r1):.3f}..{max(r1):.3f}) | {np.mean(r2):.3f} | {np.mean(r3):.3f} ({min(r3):.3f}..{max(r3):.3f}) |")
    summ[f"cos_{g}"] = {"same_seed": get(REC, SAME), "seed_pairs": r3, "rec_vs_seeds": r1, "s42_vs_seeds": r2}
P("\nStage-1 cosines for comparison (documented, from `displacement.json`):\n")
P("| pair of moves | encoder | decoder without embedding | tied embedding |")
P("|---|---:|---:|---:|")
for k, v in disp["cosines"].items():
    P(f"| {k} | {v['encoder']:.3f} | {v['decoder without embedding']:.3f} | {v['decoder embedding (tied)']:.3f} |")

P("\n### N4. Documented steps of stage 1 against the null\n")
P("Same readings on every documented step (T4, T5, T9 of `readings_tables.md`). Only `from-kurrent-v2` has the null's recipe and parent; the other steps differ in epochs, data, or parent (stated in the last column), so for them the null is a guide to the size of the noise, not an exact comparison.\n")
P("| reading | step | value | verdict against the different-seed range | step differs from the null by |")
P("|---|---|---:|---|---|")
why = {"from-kurrent-v2<-xvi-xvii": "nothing (the recorded run)", "full-v1<-xvi-xvii": "data size and epochs (450 k samples, 1 epoch)",
       "v2<-base": "parent (`base`)", "kurrent<-base": "parent, data (293 k lines), 20 epochs", "xvi-xvii<-kurrent": "parent, data, 20 epochs",
       "medieval<-base": "parent, data, 20 epochs"}
for k, lab in keys[:3] + keys[3:4] + keys[5:]:
    for step, m in doc.items():
        P(f"| {lab} | `{step}` | {fmt(m[k])} | {verdict(m[k], [nm[s][k] for s in SEEDS])} | {why[step]} |")

P("\n### N5. The five traces\n")
P("| trace | documented reading (matched step `from-kurrent-v2`) | seed null (mean, range) | verdict | same-seed repeat `s42` |")
P("|---|---|---|---|---|")
rows5 = [("encoder hump over depth", "enc_hump", "H"), ("rare token rows move most", "rows_d9_over_d0", "decile 9 / decile 0"),
         ("tied embedding dominates", "emb_share_sq", "share of squared move"), ("encoder moves far less than decoder", "enc_over_dec", "encoder / decoder")]
for lab, k, u in rows5:
    xs = [nm[s][k] for s in SEEDS]
    P(f"| {lab} | {u} {fmt(nm[REC][k])} | {fmt(np.mean(xs))} ({fmt(min(xs))}..{fmt(max(xs))}) | {verdict(nm[REC][k], xs)} | {fmt(nm[SAME][k])} |")
c = cos["encoder"]
get = lambda a, b: c.get(f"{a}|{b}", c.get(f"{b}|{a}"))
seedpairs = [get(a, b) for a, b in itertools.combinations(SEEDS, 2)]
P(f"| consecutive steps near-orthogonal (encoder cosine) | documented consecutive cosine {disp['cosines']['base->kurrent | kurrent->xvi-xvii']['encoder']:.3f} | seed pairs {np.mean(seedpairs):.3f} ({min(seedpairs):.3f}..{max(seedpairs):.3f}) | "
  f"{'consecutive cosine is inside the seed-pair range: not distinguishable' if min(seedpairs) <= disp['cosines']['base->kurrent | kurrent->xvi-xvii']['encoder'] <= max(seedpairs) else 'documented consecutive cosine lies outside the seed-pair range (lower than seeds sharing a parent)'} | {get(REC, SAME):.3f} |")

Path(REPO / "docs/trocr/nulls_tables.md").write_text("\n".join(out) + "\n")
Path(REPO / "docs/trocr/nulls_summary.json").write_text(json.dumps({"null": nm, "documented": doc, "summary": summ, "cosines": cos}, indent=1, default=float))
print("\n".join(out))
