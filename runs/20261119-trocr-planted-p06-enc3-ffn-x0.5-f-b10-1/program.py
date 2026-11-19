"""s05, task (a), budget 10, with the donor. Written in full before any pass."""
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
W = "enc.03.ffn.out.weight"
B = "enc.03.ffn.out.bias"

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

m0 = s.measure("C0", rule_mask=MASK)
s.note(f"C0 cer={m0.cer:.4f} collapsed={m0.collapsed}")

# H1: enc.03 FFN output projection was halved -> scale back by 2
h1 = s.hypothesis(
    "enc.03.ffn.out weight and bias were multiplied by 0.5 (norm ratio to parent 0.503 / 0.500, cosine 0.996 / 1.000); scale by 2.0",
    "CER falls from about 0.235 to 0.05-0.10 (drop 0.13-0.18); collapsed lines to 0-1; space deletions below 100",
    "scale",
)
s.edit("scale", [W, B], hypothesis=h1, factor=2.0)
m1 = s.measure("H1 enc.03.ffn.out x2", rule_mask=MASK)
kept = s.keep("H1: undo the halving of enc.03.ffn.out")
s.note(f"H1 cer={m1.cer:.4f} collapsed={m1.collapsed} kept={kept}")

if not kept:
    # H1b: bias from the parent, weight rescaled to the parent's norm
    P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
    T = s.subject.tensors()
    w = np.array(T[W], dtype=np.float32)
    f = float(np.linalg.norm(P[W]) / max(float(np.linalg.norm(w)), 1e-12))
    h1b = s.hypothesis(
        "enc.03.ffn.out: weight rescaled to the parent's norm, bias taken from the parent",
        "same as H1 within noise: CER drop of 0.13-0.18",
        "transplant",
    )
    s.edit("transplant", [W, B], hypothesis=h1b,
           per_tensor={W: {"donor": (w * f).astype(np.float32)},
                       B: {"donor": np.array(P[B], dtype=np.float32)}})
    m2 = s.measure("H1b enc.03.ffn.out to parent norm", rule_mask=MASK)
    kept = s.keep("H1b: enc.03.ffn.out at the parent's norm")
    s.note(f"H1b cer={m2.cer:.4f} collapsed={m2.collapsed} kept={kept} factor={f:.4f}")
    last = m2
else:
    last = m1

if kept and last.cer > 0.15:
    s.note("H2: CER still above 0.15 after the repair; a second defect may exist that the norm/cosine readings do not locate. No further edit.")
else:
    s.note("H2: no further anomaly in the readings; no further edit.")

summary = s.freeze()
