"""planted-f1 s07, task (b), budget 0: the blind programmer's program.

Reading (hypotheses.md): the L_16 BiLSTM block is at ~half the scale of L_12/L_14 in every trainable
tensor (weights and biases alike), while the unrelated donor has L_16 ~= L_14. The practice reading is
deletion-dominated (length ratio 0.83, 24 collapsed lines). One edit: scale L_16 by 2.0, guarded by a
free check on the tensors themselves. No pass, no donor, no measure.
"""
from wsp.run import open_session
import numpy as np

s = open_session(".")
T = s.subject.tensors()

L16 = ["L_16.layer.weight_ih_l0", "L_16.layer.weight_hh_l0", "L_16.layer.bias_hh_l0",
       "L_16.layer.weight_ih_l0_reverse", "L_16.layer.weight_hh_l0_reverse", "L_16.layer.bias_hh_l0_reverse"]
L14 = [n.replace("L_16", "L_14") for n in L16]


def rms(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a)))


ratios = [rms(T[a]) / rms(T[b]) for a, b in zip(L16, L14)]
ratio = float(np.mean(ratios))
s.note("rms ratio L_16/L_14 per tensor (ih, hh, bias; fwd then rev): "
       + ", ".join(f"{r:.3f}" for r in ratios) + f"; mean {ratio:.3f}")

h = s.hypothesis(
    "The L_16 BiLSTM block (weights and biases, both directions) was scaled by ~0.5: its rms is ~0.5x "
    "that of the identically built L_12/L_14, the unrelated donor has L_16 ~= L_14, and the practice "
    "reading is deletion-dominated (blank wins against damped logits).",
    "CER falls strongly, 0.199 -> ~0.03-0.08; length ratio back near 1.0; collapsed lines near 0; "
    "the deletion confusions mostly disappear.",
    "scale",
)

if 0.40 <= ratio <= 0.60:
    s.edit("scale", L16, hypothesis=h, factor=2.0)
    s.keep_unmeasured("budget 0: L_16/L_14 rms ratio %.3f is in [0.40, 0.60]; undo the x0.5 of the block "
                      "by x2.0 (hypotheses.md H1)" % ratio)
else:
    s.note("guard failed: L_16/L_14 rms ratio %.3f outside [0.40, 0.60]; nothing applied" % ratio)

s.freeze()
