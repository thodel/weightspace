"""program_02 — round 2 of 20261124-bern-combined-v1-f2-b10-2.

Kept state on entry: 0.75 subject + 0.25 parent in every tensor (H2), blank bias -0.7 (H3).
Localisation of the parent step (family F2 of the pre-registration), two passes:
  L1 (pass 5): late block (L_16, O_18) set back to the unedited subject's values; blank bias untouched.
  L2 (pass 6): if L1 is kept   -> L2a: front and middle (C_*, L_12, L_14) a further step toward the parent (alpha 0.2);
               if L1 is not kept -> L2b: front block (C_*, L_12) set back to the unedited subject's values.
No freeze here.
"""
import json
import sys
from pathlib import Path

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows
from wsp.metrics import per_line_edits, bootstrap_delta

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"

s = open_session(".")
print("budget_left at start:", s.budget_left)
if s.budget_left != 6:
    print("STOP: program_02 expects 6 passes left; nothing done.")
    sys.exit(0)

T = s.subject.tensors()
cm = s.subject.codec_map()
refs = [it["text"] for it in s.practice]
is_h = [("Hertwig" in it["key"]) for it in s.practice]
ALL = [n for n in T if "bias_ih" not in n]
FRONT = [n for n in ALL if n.startswith(("C_0.", "C_3.", "C_6.", "C_9.", "L_12."))]
MID = [n for n in ALL if n.startswith("L_14.")]
LATE = [n for n in ALL if n.startswith(("L_16.", "O_18."))]
assert len(FRONT) == 14 and len(MID) == 6 and len(LATE) == 8, (len(FRONT), len(MID), len(LATE))


def read_lines(n):
    rows = [json.loads(l) for l in Path(f"readings/pass_{n:03d}.jsonl").read_text().splitlines() if l.strip()]
    assert [r["key"] for r in rows] == [it["key"] for it in s.practice]
    return per_line_edits([r["text"] for r in rows], refs)


def cer(lines, flag=None):
    sel = [l for l, h in zip(lines, is_h) if flag is None or h == flag]
    return sum(e for e, _ in sel) / sum(c for _, c in sel)


def hand_delta(before, after):
    out = {}
    for name, flag in (("Hertwig", True), ("Schlussred", False)):
        b = [l for l, h in zip(before, is_h) if h == flag]
        a = [l for l, h in zip(after, is_h) if h == flag]
        d = bootstrap_delta(b, a, resamples=2000, seed=20261106)
        out[name] = (round(d["delta_cer"], 4), [round(x, 4) for x in d["ci95"]])
    return out


lines_c0 = read_lines(1)
bad0 = [c > 0 and e > 0.5 * c for e, c in lines_c0]     # the 40 lines that fail wholesale at C0


def groups(lines):
    out = {}
    for name, f in (("failed@C0 H", lambda i: bad0[i] and is_h[i]), ("failed@C0 S", lambda i: bad0[i] and not is_h[i]),
                    ("ok@C0 H", lambda i: not bad0[i] and is_h[i]), ("ok@C0 S", lambda i: not bad0[i] and not is_h[i])):
        out[name] = sum(lines[i][0] for i in range(len(lines)) if f(i))
    return out


kept_lines = read_lines(4)                               # pass 4 (H3) is the last kept state
kept_collapsed = 18
print(f"kept state on entry: CER {cer(kept_lines):.4f} (H {cer(kept_lines, True):.4f}, S {cer(kept_lines, False):.4f}); "
      f"edits by group {groups(kept_lines)}; blank bias {float(T['O_18.lin.bias'][0]):+.4f}")
if abs(cer(kept_lines) - 0.1346) > 5e-4 or abs(float(T["O_18.lin.bias"][0]) - 0.268) > 0.01:
    print("STOP: the restored state is not the kept state of round 1; nothing done.")
    sys.exit(0)


def report(label, m, lines):
    print(f"\n=== {label} ===")
    print(f"CER {m.cer:.4f}  edits {m.char_edits}/{m.ref_chars}  collapsed {m.collapsed}  "
          f"length_ratio {m.length_ratio:.3f}  WER {m.wer:.4f}")
    print(f"per hand: Hertwig {cer(lines, True):.4f}  Schlussred {cer(lines, False):.4f}")
    print("edits by group:", groups(lines), "| kept state:", groups(kept_lines))
    print("confusions:", m.confusions)
    e = s._log[-1]
    vk, vb = e["vs_kept"], e["vs_baseline"]
    print(f"vs_kept delta {vk['delta_cer']:+.4f} ci95 [{vk['ci95'][0]:+.4f}, {vk['ci95'][1]:+.4f}]   "
          f"vs_baseline delta {vb['delta_cer']:+.4f} ci95 [{vb['ci95'][0]:+.4f}, {vb['ci95'][1]:+.4f}]")
    print("per-hand delta vs kept (delta, ci95):", hand_delta(kept_lines, lines))
    print("lines better", sum(1 for a, b in zip(lines, kept_lines) if a[0] < b[0]),
          "worse", sum(1 for a, b in zip(lines, kept_lines) if a[0] > b[0]))


def decide(label, m, lines):
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


def with_current_blank(name, arr):
    """The blank bias (H3) is not part of these edits: its entry keeps the current value."""
    if name == "O_18.lin.bias":
        arr = arr.copy()
        arr[0] = T[name][0]
    return arr


# the unedited subject, read-only, as the source of its own original values (same codec, same rows)
orig = KrakenSubject(MODELS + "kraken-bern-combined-v1.mlmodel")
assert orig.codec_map() == cm
O = {n: orig.tensors()[n].copy() for n in ALL}
del orig

# ---------------------------------------------------------------- L1, pass 5
h = s.hypothesis(
    "L1: on the lines that were readable at C0, the parent step H2 gained 100 edits on Hertwig and cost 77 on "
    "Schlussredaktion; the cost shows as conventions of the parent (D->d 12->27, '/' and space errors). If such "
    "conventions sit in the late block and the Hertwig gain in the visual front, then setting L_16 and O_18 back to "
    "the unedited subject's values (blank bias left at its kept value) removes most of the Schlussredaktion cost and "
    "keeps most of the Hertwig gain. The parent never saw Schlussredaktion, so that cost is not recall and should "
    "also exist on held-out; the Hertwig gain may be recall of the parent's training lines.",
    "practice CER -0.2 points against the kept state (range -0.8 to +0.6); Schlussredaktion -0.3 to -0.8, "
    "Hertwig +0.3 to +1.5; collapsed 18 +/- 2; probability that the rule keeps it about 0.25",
    "transplant of original subject values into a block (L_16, O_18), i.e. localisation of the parent step")
s.edit("transplant", LATE, hypothesis=h, per_tensor={n: {"donor": with_current_blank(n, O[n])} for n in LATE})
m = s.measure("L1 late block back to subject")
lines = read_lines(5)
report("L1 late block (L_16, O_18) back to subject (pass 5)", m, lines)
k1 = decide("L1 late block back to subject", m, lines)

# ---------------------------------------------------------------- L2, pass 6
if k1:
    par = KrakenSubject(MODELS + "kraken-bern-hertwig-v2.mlmodel")
    P = par.tensors()
    names = FRONT + MID                                  # no output layer here, so no row alignment is needed
    for n in names:
        assert P[n].shape == T[n].shape, n
    h = s.hypothesis(
        "L2a: L1 was kept, so the Schlussredaktion cost of the parent step sits in the late block and the front "
        "carries the Hertwig gain. A further step of the conv stack, L_12 and L_14 toward the parent (from 0.25 to "
        "0.40 parent) gains more on Hertwig at little cost on Schlussredaktion. The parent was trained on all 103 "
        "Hertwig practice lines.",
        "practice CER -0.2 points against the kept state (range -0.8 to +0.5); Hertwig -1 to -2, "
        "Schlussredaktion 0 to +0.6; probability that the rule keeps it about 0.3",
        "interpolate toward donor (parent), blocks C_0..C_9, L_12, L_14, alpha 0.2")
    s.edit("interpolate", names, hypothesis=h, per_tensor={n: {"other": P[n]} for n in names}, alpha=0.2)
    label = "L2a front+middle toward parent a=0.2"
else:
    h = s.hypothesis(
        "L2b: L1 was not kept, so the late block is not where the Schlussredaktion cost can be removed for free. "
        "The mirror test: the conv stack and L_12 set back to the unedited subject's values, the parent share kept "
        "in L_14, L_16 and O_18. If the Hertwig gain lives in the recurrent top and the Schlussredaktion cost in "
        "the visual front, CER falls.",
        "practice CER -0.1 points against the kept state (range -0.6 to +0.8); Schlussredaktion -0.2 to -0.6, "
        "Hertwig +0.3 to +2; probability that the rule keeps it about 0.2",
        "transplant of original subject values into a block (C_0..C_9, L_12), i.e. localisation of the parent step")
    s.edit("transplant", FRONT, hypothesis=h, per_tensor={n: {"donor": O[n]} for n in FRONT})
    label = "L2b front block back to subject"
m = s.measure(label)
lines = read_lines(6)
report(label + " (pass 6)", m, lines)
k2 = decide(label, m, lines)

print("\nkept in this program: L1", k1, "|", label, k2)
print("blank bias now:", float(T["O_18.lin.bias"][0]), "| budget_left:", s.budget_left)
