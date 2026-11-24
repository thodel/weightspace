"""program_01 — round 1 of 20261124-bern-combined-v1-f2-b10-2.

Pass 1: C0. Then, each on the last kept state:
  H1 (pass 2): whole-network average with the sibling (interpolate, alpha 0.5).
  H2 (pass 3): whole-network step toward the parent (interpolate, alpha 0.25).
  H3 (pass 4): blank bias lowered; size chosen by the pre-registered rule from the C0 readings
               (or skipped, which leaves the pass unspent).
No freeze here.
"""
import json
import sys
from pathlib import Path

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows
from wsp.metrics import per_line_edits, bootstrap_delta, align, normalise

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"
OUT = ("O_18.lin.weight", "O_18.lin.bias")

s = open_session(".")
print("budget_left at start:", s.budget_left)
if s.budget_left != 10:
    print("STOP: program_01 expects a fresh session (10 passes left); nothing done.")
    sys.exit(0)

T = s.subject.tensors()
cm = s.subject.codec_map()
refs = [it["text"] for it in s.practice]
is_h = [("Hertwig" in it["key"]) for it in s.practice]
NAMES = [n for n in T if "bias_ih" not in n]          # bias_ih is all zero in every model


def read_pass():
    """Hypotheses of the pass just measured, from this run's own readings file."""
    n = s.budget - s.budget_left
    rows = [json.loads(l) for l in Path(f"readings/pass_{n:03d}.jsonl").read_text().splitlines() if l.strip()]
    assert [r["key"] for r in rows] == [it["key"] for it in s.practice]
    return [r["text"] for r in rows]


def by_hand(lines):
    out = {}
    for name, flag in (("Hertwig", True), ("Schlussred", False)):
        sel = [l for l, h in zip(lines, is_h) if h == flag]
        out[name] = sum(e for e, _ in sel) / sum(c for _, c in sel)
    return out


def hand_delta(before, after):
    out = {}
    for name, flag in (("Hertwig", True), ("Schlussred", False)):
        b = [l for l, h in zip(before, is_h) if h == flag]
        a = [l for l, h in zip(after, is_h) if h == flag]
        d = bootstrap_delta(b, a, resamples=2000, seed=20261106)
        out[name] = (round(d["delta_cer"], 4), [round(x, 4) for x in d["ci95"]])
    return out


def report(label, m, lines, kept_lines=None):
    print(f"\n=== {label} ===")
    print(f"CER {m.cer:.4f}  edits {m.char_edits}/{m.ref_chars}  collapsed {m.collapsed}  "
          f"length_ratio {m.length_ratio:.3f}  WER {m.wer:.4f}")
    print("per hand:", {k: round(v, 4) for k, v in by_hand(lines).items()})
    print("confusions:", m.confusions)
    e = s._log[-1]
    if e.get("kind") == "pass":
        vk, vb = e["vs_kept"], e["vs_baseline"]
        print(f"vs_kept delta {vk['delta_cer']:+.4f} ci95 [{vk['ci95'][0]:+.4f}, {vk['ci95'][1]:+.4f}]   "
              f"vs_baseline delta {vb['delta_cer']:+.4f} ci95 [{vb['ci95'][0]:+.4f}, {vb['ci95'][1]:+.4f}]")
    if kept_lines is not None:
        print("per-hand delta vs kept (delta, ci95):", hand_delta(kept_lines, lines))
        better = sum(1 for a, b in zip(lines, kept_lines) if a[0] < b[0])
        worse = sum(1 for a, b in zip(lines, kept_lines) if a[0] > b[0])
        print(f"lines better {better}, worse {worse}")


def aligned(donor):
    """Donor tensors in the subject's row order; rows the donor lacks keep the subject's CURRENT values."""
    dt = donor.tensors()
    rm = codec_row_map(cm, donor.codec_map())
    rm[0] = 0                                           # the CTC blank is row 0 in every model
    out = {}
    for n in NAMES:
        if n in OUT:
            out[n] = gather_rows(dt[n], rm, T[n].shape[0], fill=T[n])
        else:
            out[n] = dt[n]
        assert out[n].shape == T[n].shape, n
    return out, len(rm)


# ---------------------------------------------------------------- pass 1: C0
m0 = s.measure("C0")
hyp0 = read_pass()
lines0 = per_line_edits(hyp0, refs)
report("C0 (pass 1)", m0, lines0)
if abs(m0.cer - 0.1456) > 5e-4:
    print("STOP: C0 does not reproduce c0_c1.md (CER 0.1456). No edit made.")
    s.note(f"C0 mismatch: measured {m0.cer:.4f}, expected 0.1456; program stopped before any edit")
    sys.exit(0)

# error kinds outside the lines that fail wholesale (more than half the reference wrong)
D = I = S = 0
bad = 0
for h, r, (e, c) in zip(hyp0, refs, lines0):
    if c and e > 0.5 * c:
        bad += 1
        continue
    for a, b in align(normalise(r), normalise(h)):
        if a != b:
            if b == "":
                D += 1
            elif a == "":
                I += 1
            else:
                S += 1
print(f"\nC0: {bad} lines with more than half the reference wrong; in the other lines: "
      f"deletions {D}, insertions {I}, substitutions {S}")

kept_lines = lines0
kept_collapsed = m0.collapsed


def decide(label, m, lines):
    """Harness rule plus the collapsed rule. Returns whether the edit was kept."""
    global kept_lines, kept_collapsed
    if m.collapsed > kept_collapsed:
        s.revert(f"{label}: collapsed rose {kept_collapsed} -> {m.collapsed}")
        print(f"DECISION {label}: reverted (collapsed rose {kept_collapsed} -> {m.collapsed})")
        return False
    kept = s.keep(label)
    print(f"DECISION {label}: {'KEPT' if kept else 'refused by the keep rule, undone'}")
    if kept:
        kept_lines, kept_collapsed = lines, m.collapsed
    return kept


# ---------------------------------------------------------------- H1, pass 2: average with the sibling
sib = KrakenSubject(MODELS + "kraken-bern-schlussredaktion-v1.mlmodel")
d_sib, n_rows = aligned(sib)
print(f"\nsibling rows aligned by character: {n_rows} of {T['O_18.lin.bias'].shape[0]}")
h1 = s.hypothesis(
    "H1: subject and sibling are two fine-tunes of the same parent (task vectors cosine 0.52; output layer 0.85-0.97). "
    "Their average (all tensors, output rows aligned by character, row of 'ƒ' unchanged) reads Schlussredaktion better "
    "than either and loses a little on Hertwig. The sibling was trained on the 197 Schlussredaktion practice lines "
    "(clean) and on no Hertwig line.",
    "practice CER -0.3 points (range -1.0 to +0.5); Schlussredaktion -0.5 to -1.0, Hertwig +0.5 to +1.5; "
    "collapsed unchanged within 2; probability that the rule keeps it about 0.35",
    "interpolate toward donor (sibling), whole network, alpha 0.5")
s.edit("interpolate", NAMES, hypothesis=h1, per_tensor={n: {"other": d_sib[n]} for n in NAMES}, alpha=0.5)
m = s.measure("H1 sibling average a=0.5")
lines = per_line_edits(read_pass(), refs)
report("H1 sibling average alpha 0.5 (pass 2)", m, lines, kept_lines)
k1 = decide("H1 sibling average a=0.5", m, lines)
del sib, d_sib

# ---------------------------------------------------------------- H2, pass 3: step toward the parent
par = KrakenSubject(MODELS + "kraken-bern-hertwig-v2.mlmodel")
d_par, n_rows = aligned(par)
print(f"\nparent rows aligned by character: {n_rows} of {T['O_18.lin.bias'].shape[0]}")
h2 = s.hypothesis(
    "H2: the subject's Hertwig deficit (0.205 against the parent's 0.114) is drift away from the parent, not a "
    "localised defect: the sibling, which saw no Hertwig at all, reads Hertwig at 0.236. A quarter step back along "
    "the line to the parent (all tensors, rows aligned by character, the three new rows unchanged) recovers part of "
    "Hertwig at a small cost on Schlussredaktion. The parent was trained on all 103 Hertwig practice lines, so a "
    "Hertwig gain here may be recall of those lines and need not hold on held-out. State before this edit: "
    + ("subject averaged with sibling (H1 kept)." if k1 else "unedited subject (H1 not kept)."),
    "practice CER -0.4 points (range -1.2 to +0.6); Hertwig -2 to -4 points, Schlussredaktion +0.5 to +1.5; "
    "collapsed unchanged or lower; probability that the rule keeps it about 0.35",
    "interpolate toward donor (parent), whole network, alpha 0.25")
s.edit("interpolate", NAMES, hypothesis=h2, per_tensor={n: {"other": d_par[n]} for n in NAMES}, alpha=0.25)
m = s.measure("H2 toward parent a=0.25")
lines = per_line_edits(read_pass(), refs)
report("H2 toward parent alpha 0.25 (pass 3)", m, lines, kept_lines)
k2 = decide("H2 toward parent a=0.25", m, lines)
del par, d_par

# ---------------------------------------------------------------- H3, pass 4: blank bias
if D >= 2 * I:
    step = -0.7
elif D > I:
    step = -0.3
else:
    step = None
print(f"\nH3 rule: deletions {D}, insertions {I} outside failed lines -> blank-bias step {step}")
if step is None:
    s.note(f"H3 skipped by its pre-registered rule: deletions {D} <= insertions {I} outside failed lines")
    print("H3 skipped; pass 4 not spent.")
else:
    b0 = float(T["O_18.lin.bias"][0])
    h3 = s.hypothesis(
        f"H3: outside the lines that fail wholesale the subject deletes more than it inserts ({D} against {I} at C0), "
        f"and its blank bias is the highest of the family. Lowering the blank bias from {b0:+.3f} by {abs(step)} lets "
        "characters win the frames where the blank wins narrowly. No donor is used.",
        "practice CER -0.3 points (range -0.8 to +0.3), both hands alike; deletions down, insertions up, "
        "length ratio up by 0.005 to 0.02; collapsed unchanged; probability that the rule keeps it about 0.4",
        "add on one bias entry (blank, row 0 of O_18.lin.bias)")
    delta = np.zeros_like(T["O_18.lin.bias"])
    delta[0] = step
    s.edit("add", ["O_18.lin.bias"], hypothesis=h3, delta=delta)
    m = s.measure(f"H3 blank bias {step:+.1f}")
    lines = per_line_edits(read_pass(), refs)
    report(f"H3 blank bias {step:+.1f} (pass 4)", m, lines, kept_lines)
    k3 = decide(f"H3 blank bias {step:+.1f}", m, lines)

print("\nkept so far: H1", k1, "| H2", k2, "| H3", (k3 if step is not None else "skipped"))
print("blank bias now:", float(T["O_18.lin.bias"][0]), "| budget_left:", s.budget_left)
