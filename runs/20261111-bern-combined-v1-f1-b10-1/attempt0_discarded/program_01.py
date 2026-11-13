"""program_01 — round 0 of session 20261111-bern-combined-v1-f1-b10-1 (programmer: Claude Fable 5.1).

Pass 1: C0 baseline. Then H1 (blank bias -0.4), H1b (a second -0.4, if H1 kept; or H1b' -0.8 in
one step if H1 was refused with a negative point estimate), H2 (whole-network interpolation
toward the sibling, alpha 0.5, output rows aligned by character, blank bias held), H2b (alpha
0.25, if H2 was refused with a negative point estimate). At most 5 passes. Does NOT freeze:
follow-ups come in program_02.py after reading these results. See preregistration.md section 5.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

SIBLING = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-schlussredaktion-v1.mlmodel"
C0_CER, C0_COLLAPSED = 0.1456, 29

s = open_session(".")
if s.budget_left != 10:
    print(f"budget_left is {s.budget_left}, not 10: this program is for a fresh session; nothing done")
    raise SystemExit(0)

hert = np.array([it["key"].startswith("A_V_1443_Hertwig") for it in s.practice])
state = {"kept_collapsed": None, "kept_cer": None}


def report(label, m):
    """Print the pass: total / per-hand CER, interval vs last kept, collapsed, length ratio, confusions."""
    lines = np.array(s._last_lines, dtype=float)      # (edits, ref_chars) per practice line, this pass
    h = lines[hert, 0].sum() / lines[hert, 1].sum()
    sr = lines[~hert, 0].sum() / lines[~hert, 1].sum()
    vk = s._log[-1]["vs_kept"]
    print(f"[{label}] CER {m.cer:.4f}  Hertwig {h:.4f}  Schlussred {sr:.4f}  collapsed {m.collapsed}  "
          f"len_ratio {m.length_ratio:.3f}  edits {m.char_edits}  WER {m.wer:.4f}")
    print(f"        vs kept: delta {vk['delta_cer']:+.4f}  ci95 [{vk['ci95'][0]:+.4f}, {vk['ci95'][1]:+.4f}]")
    print("        confusions:", " ".join(f"{a or '∅'}→{b or '∅'}:{n}" for a, b, n in m.confusions))
    ins = sum(n for a, b, n in m.confusions if a == "")
    dele = sum(n for a, b, n in m.confusions if b == "")
    print(f"        top-10 deletions {dele}, insertions {ins}; blocks: "
          + "; ".join(f"{k}: {v}" for k, v in list(m.blocks.items())[:4]))
    return h, sr


def decide(m, label, extra_guard_fail=False):
    """Keep under the harness rule unless the collapsed count rose (my extra rule) or a guard failed."""
    ok, why = s.keep_allowed()
    if m.collapsed > state["kept_collapsed"]:
        s.revert(f"{label}: collapsed rose {state['kept_collapsed']} -> {m.collapsed} (programmer's rule)")
        print(f"        REVERT (collapsed rose; harness rule said {ok}, {why!r})")
        return False
    if extra_guard_fail:
        s.revert(f"{label}: programmer's guard failed")
        print(f"        REVERT (guard; harness rule said {ok}, {why!r})")
        return False
    kept = s.keep(f"{label}: interval below 0" if ok else f"{label}: requested")
    if kept:
        state["kept_collapsed"] = m.collapsed
        state["kept_cer"] = m.cer
        print("        KEPT")
    else:
        print(f"        refused by the rule ({why})")
    return kept


# ---------------------------------------------------------------- pass 1: C0
m0 = s.measure("C0")
state["kept_collapsed"], state["kept_cer"] = m0.collapsed, m0.cer
h0, s0 = report("C0", m0)
if abs(m0.cer - C0_CER) > 0.0005 or m0.collapsed != C0_COLLAPSED:
    print(f"C0 does not reproduce c0_c1.md (expected CER {C0_CER}, collapsed {C0_COLLAPSED}); stopping without an edit")
    raise SystemExit(0)
print("C0 reproduced.", s.budget_left, "passes left")

T = s.subject.tensors()
cm = s.subject.codec_map()
BLANK = 0
assert "O_18.lin.bias" in T and T["O_18.lin.bias"].shape[0] == 270

# ---------------------------------------------------------------- H1: blank bias -0.4
def blank_delta(d):
    delta = np.zeros_like(T["O_18.lin.bias"])
    delta[BLANK] = d
    return delta


h1 = s.hypothesis(
    "H1: the CTC blank bias rose by +0.118 parent->subject (more than in any other fine-tune of the family); "
    "the subject's errors are deletion-dominated (1033 deletions vs 153 insertions). Lower O_18.lin.bias[0] by 0.4.",
    "CER -0.3 to -0.8 point (0.1456 -> 0.138-0.143), both hands down proportionally; deletions of ' e n /' fall "
    "10-25 %, insertions rise by less than half of that; length ratio 0.918 -> 0.93-0.94; collapsed stays 29.",
    "bias shift (O_18.lin.bias row 0)")
s.edit("add", ["O_18.lin.bias"], hypothesis=h1, delta=blank_delta(-0.4))
print(f"blank bias now {s.subject.tensors()['O_18.lin.bias'][BLANK]:+.4f}")
m1 = s.measure("H1 blank -0.4")
report("H1 blank -0.4", m1)
h1_delta = s._log[-1]["vs_kept"]["delta_cer"]
h1_kept = decide(m1, "H1 blank -0.4")
h1_gain = state["kept_cer"] - m0.cer if h1_kept else 0.0

# ---------------------------------------------------------------- H1b / H1b'
if h1_kept:
    h1b = s.hypothesis(
        "H1b: a second blank step, -0.4 more (-0.8 in all), from the kept -0.4 state.",
        "a further -0.1 to -0.4 point, smaller than H1's gain; insertions now rise faster; collapsed not up. "
        "Refused => the optimum lies between -0.4 and -0.8, stop the blank sweep.",
        "bias shift (O_18.lin.bias row 0)")
    s.edit("add", ["O_18.lin.bias"], hypothesis=h1b, delta=blank_delta(-0.4))
    print(f"blank bias now {s.subject.tensors()['O_18.lin.bias'][BLANK]:+.4f}")
    m1b = s.measure("H1b blank -0.8")
    report("H1b blank -0.8", m1b)
    h1b_kept = decide(m1b, "H1b blank -0.8")
    if h1b_kept:
        print(f"H1b gain {m1.cer - m1b.cer:+.4f} vs H1 gain {-h1_gain:+.4f}; "
              f"H1c (third step) {'allowed' if (m1.cer - m1b.cer) >= 0.5 * (-h1_gain) else 'not allowed'} as a follow-up")
elif h1_delta < 0:
    h1bp = s.hypothesis(
        "H1b': H1 was refused but its point estimate was negative; one attempt at -0.8 in a single step.",
        "CER -0.3 to -0.8 point against C0 with an interval below 0; collapsed not up. "
        "Refused => the blank prior is not the obstacle; no further blank edit.",
        "bias shift (O_18.lin.bias row 0)")
    s.edit("add", ["O_18.lin.bias"], hypothesis=h1bp, delta=blank_delta(-0.8))
    print(f"blank bias now {s.subject.tensors()['O_18.lin.bias'][BLANK]:+.4f}")
    m1bp = s.measure("H1b' blank -0.8")
    report("H1b' blank -0.8", m1bp)
    decide(m1bp, "H1b' blank -0.8")
else:
    s.note("H1 refused with a non-negative point estimate: the blank prior is not the obstacle; no H1b.")
    print("H1 refused with delta >= 0: skipping H1b")

# ---------------------------------------------------------------- H2: interpolate toward the sibling
sib = KrakenSubject(SIBLING)
TS = sib.tensors()
cs = sib.codec_map()
T = s.subject.tensors()


def sibling_arrays():
    """Donor arrays in the subject's layout: equal-shape tensors as they are; O_18 rows by character;
    the subject's own row for a character the sibling lacks (ƒ); the blank bias held at its current value."""
    out = {}
    for n, w in T.items():
        if n.startswith("O_18"):
            continue
        if TS[n].shape == w.shape:
            out[n] = TS[n]
    w_other = T["O_18.lin.weight"].copy()
    b_other = T["O_18.lin.bias"].copy()
    w_other[BLANK] = TS["O_18.lin.weight"][BLANK]          # blank row weights: interpolate
    n_aligned = 0
    for ch, i in cm.items():
        j = cs.get(ch)
        if j is None:
            continue
        w_other[i] = TS["O_18.lin.weight"][j]
        b_other[i] = TS["O_18.lin.bias"][j]
        n_aligned += 1
    # b_other[BLANK] stays the subject's current (kept) blank bias: H2 does not undo H1
    out["O_18.lin.weight"] = w_other
    out["O_18.lin.bias"] = b_other
    print(f"sibling donor arrays: {len(out)} tensors; {n_aligned} output rows aligned by character; "
          f"rows kept from the subject: {[c for c in cm if c not in cs]}")
    return out


donor = sibling_arrays()
names = list(donor)
h2 = s.hypothesis(
    "H2: whole-network interpolation toward the sibling (schlussredaktion-v1), alpha 0.5, output rows aligned by "
    "character, blank bias held. Donor: trained on the 197 Schlussredaktion practice lines (as the subject was, "
    "correct images in both), never on Hertwig; the least confounded donor. Halves the component the subject "
    "learnt from the mismatched Hertwig pages (subject - sibling, both children of hertwig-v2).",
    "CER -0.3 to -1.0 point vs last kept; Schlussredaktion -0.3 to -0.8 (soup of two readers of this hand); "
    "Hertwig between -0.5 and +1.5; collapsed not up. Refused with delta<0 => try alpha 0.25 (H2b); "
    "refused with delta>=0 => not linearly connected / noisy component carries real Hertwig ability; leave axis.",
    "interpolation (all tensors, alpha 0.5)")
s.edit("interpolate", names, hypothesis=h2, alpha=0.5, per_tensor={n: {"other": donor[n]} for n in names})
print(f"blank bias now {s.subject.tensors()['O_18.lin.bias'][BLANK]:+.4f}")
m2 = s.measure("H2 sibling alpha 0.5")
report("H2 sibling alpha 0.5", m2)
h2_delta = s._log[-1]["vs_kept"]["delta_cer"]
h2_kept = decide(m2, "H2 sibling alpha 0.5")

if not h2_kept and h2_delta < 0:
    T = s.subject.tensors()
    donor = sibling_arrays()               # rebuilt from the restored state (blank bias as kept)
    h2b = s.hypothesis(
        "H2b: H2 (alpha 0.5) was refused with a negative point estimate; same interpolation at alpha 0.25.",
        "CER -0.1 to -0.5 point vs last kept, Schlussredaktion down, Hertwig about flat; collapsed not up. "
        "Refused => leave the sibling axis.",
        "interpolation (all tensors, alpha 0.25)")
    s.edit("interpolate", names, hypothesis=h2b, alpha=0.25, per_tensor={n: {"other": donor[n]} for n in names})
    m2b = s.measure("H2b sibling alpha 0.25")
    report("H2b sibling alpha 0.25", m2b)
    decide(m2b, "H2b sibling alpha 0.25")
elif not h2_kept:
    s.note("H2 refused with a non-negative point estimate: leaving the sibling-interpolation axis.")
    print("H2 refused with delta >= 0: skipping H2b")

print(f"\nend of program_01: {s.budget_left} passes left; kept CER {state['kept_cer']:.4f}, "
      f"kept collapsed {state['kept_collapsed']}; blank bias {s.subject.tensors()['O_18.lin.bias'][BLANK]:+.4f}")
print("not frozen: follow-ups in program_02.py")
