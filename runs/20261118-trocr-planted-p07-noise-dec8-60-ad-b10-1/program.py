from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

NAMES = ["dec.08.%s.weight" % n for n in
         ["self.q", "self.k", "self.v", "self.o", "cross.q", "cross.k", "cross.v", "cross.o", "ffn.in", "ffn.out"]]
DONOR = {n: {"other": P[n].copy()} for n in NAMES}

m = s.measure("C0", rule_mask=MASK)
best = m.cer
s.note("C0 CER %.4f collapsed %s" % (m.cer, m.collapsed))


def step(alpha, text, pred, reason):
    """Interpolate dec.08 weights toward the parent by alpha (relative to the current state)."""
    global best
    h = s.hypothesis(text, pred, "interpolate")
    s.edit("interpolate", NAMES, hypothesis=h, per_tensor=DONOR, alpha=alpha)
    mm = s.measure("interp %.2f" % alpha, rule_mask=MASK)
    ok = (mm.collapsed <= m.collapsed + 2) and s.keep(reason)
    if not ok and s._last_edit is not None:
        s.revert("collapse or rule refused: " + reason)
    if ok:
        best = mm.cer
    s.note("alpha %.2f cer %.4f kept %s" % (alpha, mm.cer, ok))
    return ok


# cumulative alpha 0.5, 0.75, 0.9 ; relative alphas below
cum = 0.0
ok1 = step(0.5, "dec.08 weights carry added noise; pull halfway to the parent",
           "CER -4 to -6 points", "H1 alpha 0.5")
if ok1:
    cum = 0.5
else:
    if step(0.25, "H1 fallback: smaller pull toward the parent", "CER -1 to -3 points", "H1b alpha 0.25"):
        cum = 0.25

if cum > 0:
    # relative alpha to go from cum to target: (target-cum)/(1-cum)
    for target in (0.75, 0.9):
        rel = (target - cum) / (1.0 - cum)
        if step(rel, "further noise removal, cumulative alpha %.2f" % target,
                "CER -1 to -3 points" if target < 0.8 else "CER -0.5 to +1 point", "H2/H3 cumulative %.2f" % target):
            cum = target
        else:
            break
    # H4: restore the parent's norm per tensor (the noise inflated it)
    if cum < 1.0:
        # joint test: single common factor from the mean norm ratio
        T = s.subject.tensors()
        f = float(np.mean([np.linalg.norm(P[n]) / max(np.linalg.norm(T[n]), 1e-8) for n in NAMES]))
        h = s.hypothesis("H4 joint norm correction of dec.08 weights", "CER -0.5 to -1 point", "scale")
        s.edit("scale", NAMES, hypothesis=h, factor=f)
        mm = s.measure("norm fix %.3f" % f, rule_mask=MASK)
        if not s.keep("H4 norm correction"):
            pass

summary = s.freeze()
