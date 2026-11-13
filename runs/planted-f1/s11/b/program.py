"""Blind programmer (budget 0), planted-f1 / s11 / b. See hypotheses.md.

Two repairs, each guarded by a reading of the weights (free: no pass is needed to read tensors):
  H1  the space row's output bias, planted down to -7.85, is set back to where the parent's value plus the
      ordinary fine-tuning shift of the frequent letters puts it;
  H2  the four L_14 weight matrices, which carry isotropic noise against the parent, are interpolated 80 % of the way
      toward the parent (biases untouched).
Nothing is measured; every applied edit is kept with keep_unmeasured. Ends with freeze().
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
ALPHA = 0.8            # H2: share of the way toward the parent
SPACE_GAP = 3.0        # H1: apply only if the space bias lies more than this below its expected value
RATIO_W = 1.5          # H2: L_14 rel L2 must exceed this times the largest same-kind rel L2 of L_12/L_16
RATIO_B = 1.2          # H2: ... while L_14's bias_hh moved at most this times the neighbours' average

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
P = parent.tensors()
pm = parent.codec_map()


def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)) / np.linalg.norm(b.astype(np.float64)))


s.note("budget 0: blind programmer. Readings used: pack describe/diff/c0. Two guarded repairs, see hypotheses.md.")

# ---------------------------------------------------------------------------------------------------------------
# H1: the space row's bias (-7.85 in the pack, the most negative of all rows) -> parent value + typical shift
# ---------------------------------------------------------------------------------------------------------------
cb, pb = T["O_18.lin.bias"], P["O_18.lin.bias"]
sp, psp = cm.get(" "), pm.get(" ")
if sp is None or psp is None:
    s.note("H1 skipped: no single-label space row in child or parent codec")
else:
    ref_chars = [c for c in "enrisatdhulcgmob" if c in cm and c in pm]
    shifts = np.array([float(cb[cm[c]]) - float(pb[pm[c]]) for c in ref_chars])
    shift = float(np.median(shifts)) if len(shifts) else 0.0
    expected = float(pb[psp]) + shift
    current = float(cb[sp])
    row_rel = rel(T["O_18.lin.weight"][sp], P["O_18.lin.weight"][psp])
    s.note(f"H1 reading: space bias child {current:+.3f}, parent {float(pb[psp]):+.3f}, median shift of frequent "
           f"letters {shift:+.3f} (n={len(shifts)}), expected {expected:+.3f}; space weight row rel L2 {row_rel:.3f}")
    if current < expected - SPACE_GAP:
        h = s.hypothesis(
            "The space row's output bias was planted down (-7.85, below every never-seen character); the 156 "
            "space deletions and the WER/CER gap follow from it. Restore it to parent bias + the median "
            "child-parent shift of frequent lowercase letters.",
            "CER down ~0.008-0.011 (most of the 156 ' '->nothing deletions recovered, a few new space insertions); "
            "WER down markedly (~0.32 -> ~0.25).",
            "output-row bias repair (set_rows on O_18.lin.bias, one row)")
        s.edit("set_rows", ["O_18.lin.bias"], hypothesis=h, index=[int(sp)],
               values=np.array([expected], dtype=np.float32))
        s.keep_unmeasured(f"budget 0; space bias {current:+.2f} -> {expected:+.2f}, gap {expected - current:.2f} "
                          f"> {SPACE_GAP}")
    else:
        s.note(f"H1 skipped: space bias {current:+.3f} is within {SPACE_GAP} of its expected value {expected:+.3f}")

# ---------------------------------------------------------------------------------------------------------------
# H2: isotropic noise on the L_14 weight matrices -> interpolate toward the parent (biases untouched)
# ---------------------------------------------------------------------------------------------------------------
KINDS = ("weight_ih_l0", "weight_hh_l0")
DIRS = ("", "_reverse")
L14 = [f"L_14.layer.{k}{d}" for k in KINDS for d in DIRS]
r14 = {n: rel(T[n], P[n]) for n in L14}
rnb = {k: max(rel(T[f"L_{l}.layer.{k}{d}"], P[f"L_{l}.layer.{k}{d}"]) for l in (12, 16) for d in DIRS) for k in KINDS}
b14 = np.mean([rel(T[f"L_14.layer.bias_hh_l0{d}"], P[f"L_14.layer.bias_hh_l0{d}"]) for d in DIRS])
bnb = np.mean([rel(T[f"L_{l}.layer.bias_hh_l0{d}"], P[f"L_{l}.layer.bias_hh_l0{d}"]) for l in (12, 16) for d in DIRS])
s.note("H2 reading: L_14 rel L2 " + ", ".join(f"{n.split('.')[-1]} {v:.3f}" for n, v in r14.items())
       + f"; neighbours' max ih {rnb['weight_ih_l0']:.3f}, hh {rnb['weight_hh_l0']:.3f}; "
       f"bias_hh rel L_14 {b14:.4f} vs neighbours {bnb:.4f}")
weights_far = all(r14[f"L_14.layer.{k}{d}"] > RATIO_W * rnb[k] for k in KINDS for d in DIRS)
bias_quiet = b14 <= RATIO_B * bnb
if weights_far and bias_quiet:
    h = s.hypothesis(
        "L_14's four weight matrices carry isotropic noise (sigma ~0.15) against the parent: 2.5x the neighbours' "
        "weight change with the smallest bias change, cosines of an orthogonal perturbation, rms grown 14-25 %. "
        f"Interpolate them {ALPHA:.0%} of the way toward the parent; biases untouched.",
        "CER down 0.03-0.06 (to ~0.06-0.09); collapsed lines 11 -> ~0-3; deletions of frequent letters down. "
        "If CER rises instead, the L_14 drift was genuine.",
        "block interpolation toward the parent donor (interpolate, alpha=0.8, 4 tensors)")
    s.edit("interpolate", L14, hypothesis=h, alpha=ALPHA, per_tensor={n: {"other": P[n]} for n in L14})
    after = {n: rel(s.subject.tensors()[n], P[n]) for n in L14}
    s.keep_unmeasured("budget 0; L_14 weights rel L2 to parent now "
                      + ", ".join(f"{n.split('.')[-1]} {v:.3f}" for n, v in after.items()))
else:
    s.note(f"H2 skipped: weights_far={weights_far}, bias_quiet={bias_quiet} (thresholds {RATIO_W}, {RATIO_B})")

summary = s.freeze()
print("edits", summary["edits"], "kept", summary["kept"], "changed", sorted(summary["changed_tensors"]))
