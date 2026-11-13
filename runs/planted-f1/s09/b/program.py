"""planted-f1 s09, task (b), budget 0: blind repair of the zeroed conv block C_3.

Readings (pack): C_3.co.weight and C_3.co.bias are exactly zero (L2 0, cosine nan vs the parent), every other
tensor is at ordinary fine-tuning distance from kraken-german-xix-v1; the practice reading is CER 1.0 with all
300 lines collapsed. A Cr block with zero weight and bias outputs relu(0) = 0 on every frame, so everything
downstream sees a constant and the model emits nothing. Repair: transplant the parent's C_3 (same shapes, same
channel order). No passes exist; the edit is kept unmeasured. See hypotheses.md.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
NAMES = ["C_3.co.weight", "C_3.co.bias"]

s = open_session(".")
T = s.subject.tensors()

s.note("blind programmer (budget 0). Readings: C_3.co.weight/bias all zero (L2 0.00, rel L2 1.0, cosine nan vs parent); "
       "all other blocks at rel L2 0.13-0.33 / cosine >= 0.95 to the parent; C0 practice CER 1.0000, 300/300 lines "
       "collapsed, length ratio 0.0. Mechanism: Cr3,13,32 with zero weight and bias -> relu(0) = 0 on every frame -> "
       "constant input to C_6..O_18 -> empty output on every line.")

# Sanity checks before touching anything: the live tensors must show the defect the pack describes,
# and the parent must hold same-shaped, non-zero tensors. If not, apply nothing.
zero_here = all(float(np.abs(T[n]).max()) == 0.0 for n in NAMES)
parent = KrakenSubject(PARENT)
P = parent.tensors()
shapes_ok = all(n in P and P[n].shape == T[n].shape for n in NAMES)
parent_nonzero = shapes_ok and all(float(np.linalg.norm(P[n])) > 0.0 for n in NAMES)

if not zero_here:
    s.note("C_3 is NOT all-zero in the live tensors (max |w| = %s, max |b| = %s); the readings I planned on do not hold. "
           "Applying nothing." % (float(np.abs(T[NAMES[0]]).max()), float(np.abs(T[NAMES[1]]).max())))
elif not parent_nonzero:
    s.note("parent C_3 missing, differently shaped, or zero: no donor. Applying nothing.")
else:
    h = s.hypothesis(
        "C_3 (Cr3,13,32) has been zeroed: weight [32,32,3,13] and bias [32] are exactly 0, so the block outputs "
        "relu(0)=0 everywhere and the network collapses to an empty string on every line. The rest of the model is a "
        "healthy child of kraken-german-xix-v1 (rel L2 0.13-0.33, cosine >= 0.95 in every other block), so the parent's "
        "C_3 is approximately the lost block and sits in the same channel order.",
        "CER on the 300 practice lines falls from 1.000 to roughly 0.10-0.30 (parent-level or somewhat worse, since "
        "C_6/C_9 were co-adapted to the child's own C_3); collapsed lines from 300 to near 0; length ratio from 0.0 to "
        "about 0.9-1.0. Large decrease. If CER stays near 1.0 there is a second, downstream cause the readings do not show.",
        "transplant (parent donor, whole conv block C_3: weight + bias)")
    s.edit("transplant", NAMES, hypothesis=h,
           per_tensor={n: {"donor": P[n].astype(np.float32)} for n in NAMES})
    s.keep_unmeasured("budget 0: an all-zero conv block with zero bias is a defect, not a training outcome; the practice "
                      "reading (CER 1.0, 300/300 collapsed) is exactly its signature; the parent's same-shaped C_3 is "
                      "the only donor and the only edit I am confident in without a pass")
    s.note("after the transplant C_3.co.weight L2 = %.3f, C_3.co.bias L2 = %.3f (parent values)."
           % (float(np.linalg.norm(s.subject.tensors()[NAMES[0]])), float(np.linalg.norm(s.subject.tensors()[NAMES[1]]))))

s.note("checked and not acted on: output layer blank row (norm 4.64, bias +1.23) and frequent-letter rows being the "
       "weakest are normal CTC structure; the space row bias (-7.85, most negative of all) is odd for the most frequent "
       "character but is not in the moved-most table and the collapsed practice reading gives no confusion evidence, so "
       "it is left alone; L_16.weight_ih_l0 abs max 4.70 sits in a tensor whose rel L2/cosine to the parent are in line "
       "with its siblings. No LSTM block and no other conv block shows a scale, zero or sign anomaly. No second edit.")

summary = s.freeze()
print("edits", summary["edits"], "kept", summary["kept"], "changed", sorted(summary["changed_tensors"]))
