"""planted-f1 / s04 / task (b): blind programmer, budget 0.

Two guarded repairs, decided on the weights alone (no forward pass is available):
  H1  the space row's output bias (-7.85, the most negative of all 297 classes) is restored to the
      parent's value plus this child's frequent-letter bias drift;
  H2  isolated weight elements that sit > 1.5 away from the parent (~25x the per-element drift) in
      conv/LSTM tensors are set back to the parent's values (candidate: the 4.23 in L_16 weight_ih_l0).
Each guard that does not fire writes a note instead of an edit. See hypotheses.md.
"""
import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

# frequent letters of 19th-century German text: their biases/rows are the "healthy frequent class" reference
FREQUENT = list("enrisdtauhlgcomb")

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()

parent = None
pT = None
pm = None
try:
    parent = KrakenSubject(PARENT)
    pT = parent.tensors()
    pm = parent.codec_map()
    s.note("parent loaded as donor: %s" % PARENT)
except Exception as exc:  # no parent -> child-only guards
    s.note("parent could not be loaded (%r); falling back to child-only guards" % (exc,))


def robust_z(x, ref):
    ref = np.asarray(ref, dtype=np.float64)
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) * 1.4826
    mad = max(mad, 1e-3)
    return (float(x) - med) / mad, med, mad


# ----------------------------------------------------------------------------------------------------------
# H1: space bias
# ----------------------------------------------------------------------------------------------------------
bias = T["O_18.lin.bias"]
if " " not in cm:
    s.note("H1 skipped: no single-label space row in the child codec")
else:
    sp = cm[" "]
    freq_rows = [cm[c] for c in FREQUENT if c in cm]
    child_sp = float(bias[sp])
    apply_h1 = False
    target = None
    detail = ""
    if pm is not None and " " in pm and len(freq_rows) >= 8:
        pbias = pT["O_18.lin.bias"]
        shared_freq = [(cm[c], pm[c]) for c in FREQUENT if c in cm and c in pm]
        deltas = [float(bias[ic]) - float(pbias[ip]) for ic, ip in shared_freq]
        d_sp = child_sp - float(pbias[pm[" "]])
        z, med_d, mad_d = robust_z(d_sp, deltas)
        target = float(pbias[pm[" "]]) + med_d
        detail = ("child space bias %.3f, parent %.3f, delta %.3f; frequent-letter delta median %.3f (MAD %.3f), "
                  "robust z of the space delta %.1f; target %.3f"
                  % (child_sp, float(pbias[pm[" "]]), d_sp, med_d, mad_d, z, target))
        apply_h1 = (d_sp < -3.0) and (z < -5.0)
        if not (-3.0 <= target <= 2.0):          # the parent's value is itself odd: use the child's frequent letters
            target = float(np.median(bias[freq_rows]))
            detail += "; parent-based target out of range, using child frequent-letter median %.3f" % target
    else:
        z, med_b, mad_b = robust_z(child_sp, bias[freq_rows])
        target = med_b
        detail = ("(child-only) space bias %.3f; frequent-letter bias median %.3f (MAD %.3f), robust z %.1f; target %.3f"
                  % (child_sp, med_b, mad_b, z, target))
        apply_h1 = (child_sp < -5.0) and (z < -5.0)

    if apply_h1:
        h = s.hypothesis(
            "H1: the output bias of the space row was pushed to %.2f, far below every frequent class; restoring it "
            "to the parent's value plus this child's frequent-letter drift removes most of the 124 space deletions. %s"
            % (child_sp, detail),
            "CER down 0.5-0.8 pp (0.0727 -> ~0.065-0.068), WER down 2-4 pp; space deletions fall by ~80-110, "
            "a few new space insertions",
            "set_rows (output bias, one row)")
        s.edit("set_rows", ["O_18.lin.bias"], hypothesis=h, index=[int(sp)],
               values=np.asarray([target], dtype=np.float32))
        s.keep_unmeasured("budget 0: space bias is a > 5-MAD outlier against the frequent letters; restored to %.3f" % target)
    else:
        s.note("H1 not applied: guard did not fire. " + detail)

# ----------------------------------------------------------------------------------------------------------
# H2: isolated spikes against the parent in conv / LSTM tensors
# ----------------------------------------------------------------------------------------------------------
if pT is None:
    s.note("H2 skipped: no parent to compare against")
else:
    THRESH = 1.5
    MAX_ELEMS = 200
    T = s.subject.tensors()
    for name in sorted(T):
        if name.startswith("O_18") or name not in pT or pT[name].shape != T[name].shape:
            continue
        w = T[name].astype(np.float64)
        p = pT[name].astype(np.float64)
        d = w - p
        mask = np.abs(d) > THRESH
        n = int(mask.sum())
        if n == 0:
            continue
        where = np.argwhere(mask)[:10].tolist()
        biggest = float(np.abs(d).max())
        drift_rms = float(np.sqrt(np.mean(d[~mask] ** 2))) if (~mask).any() else 0.0
        if n > MAX_ELEMS:
            s.note("H2 %s: %d elements deviate > %.1f from the parent (max %.2f) -- too many for a spike, left alone"
                   % (name, n, THRESH, biggest))
            continue
        h = s.hypothesis(
            "H2: %s has %d isolated element(s) deviating > %.1f from the parent (max |delta| %.2f; drift rms of the "
            "rest %.3f; first positions %s): planted spike(s). Setting them back to the parent's values."
            % (name, n, THRESH, biggest, drift_rms, where),
            "CER down 0.1-0.5 pp, possibly fewer collapsed lines; at worst neutral",
            "add (parent - child on the spiked elements only)")
        delta = np.zeros_like(T[name], dtype=np.float32)
        delta[mask] = (-d[mask]).astype(np.float32)
        s.edit("add", [name], hypothesis=h, delta=delta)
        s.keep_unmeasured("budget 0: %d element(s) of %s were > %.1f from the parent, ~25x the drift rms; reverted"
                          % (n, name, THRESH))

s.freeze()
