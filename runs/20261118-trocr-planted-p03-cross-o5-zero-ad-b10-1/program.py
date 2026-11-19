from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
NAMES = ["dec.05.cross.o.weight", "dec.05.cross.o.bias"]

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
DON = {n: {"other": np.array(P[n], dtype=np.float32, copy=True)} for n in NAMES}

T = s.subject.tensors()
s.note("dec.05.cross.o zero check: " + ", ".join(f"{n} absmax={float(np.abs(T[n]).max()):.4g}" for n in NAMES))

m0 = s.measure("C0", rule_mask=MASK)
s.note(f"C0 cer={m0.cer:.4f} collapsed={m0.collapsed}")


def try_alpha(alpha, text, pred):
    h = s.hypothesis(text, pred, "interpolate")
    s.edit("interpolate", NAMES, hypothesis=h, per_tensor=DON, alpha=alpha)
    m = s.measure(f"alpha{alpha}", rule_mask=MASK)
    s.note(f"alpha {alpha}: cer={m.cer:.4f} collapsed={m.collapsed}")
    return s.keep(f"alpha {alpha} dec.05.cross.o from donor"), m


kept1, m1 = try_alpha(
    1.0,
    "dec.05.cross.o weight and bias are exactly zero (distance 2.0 to donor): layer 5 ignores the encoder; copy donor tensors",
    "CER down 0.03 to 0.10, collapsed lines down",
)
if kept1:
    # a quick check that the restoration is not under-scaled; expected to be reverted
    h = s.hypothesis(
        "restored layer 5 cross output may be slightly under-scaled (child bias norms ~1.05 of donor)",
        "CER change within +-0.003, expect revert",
        "scale",
    )
    s.edit("scale", NAMES, hypothesis=h, factor=1.2)
    m = s.measure("scale1.2", rule_mask=MASK)
    s.note(f"scale 1.2: cer={m.cer:.4f} collapsed={m.collapsed}")
    s.keep("scale 1.2 on restored dec.05.cross.o")
else:
    try_alpha(
        0.6,
        "full donor tensor may clash with the child's neighbours; partial restoration of the zeroed cross output",
        "CER down 0.01 to 0.05",
    )

summary = s.freeze()
print(summary)
