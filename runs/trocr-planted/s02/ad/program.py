from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
B = "dec.embed.norm.bias"

m0 = s.measure("C0", rule_mask=MASK)
s.note(f"C0 cer={m0.cer:.4f} collapsed={m0.collapsed}")

# H1: transplant the decoder embedding LayerNorm bias from the donor
h = s.hypothesis("dec.embed.norm.bias was shifted/noised (norm 6.3x parent, cosine 0.12); restoring the donor bias lowers CER >= 0.05",
                 "CER down by about 0.05-0.10, collapsed lines 33 -> <15", "transplant")
s.edit("transplant", [B], hypothesis=h, per_tensor={B: {"donor": P[B]}})
m1 = s.measure("H1 transplant embed norm bias", rule_mask=MASK)
kept1 = s.keep("embedding LN bias restored from donor")

if not kept1:
    # H2: remove only a constant offset
    T = s.subject.tensors()
    shift = float(np.mean(T[B] - P[B]))
    delta = np.full_like(T[B], -shift)
    h = s.hypothesis("the defect is a mostly constant offset on the embedding LN bias; removing the mean offset restores reading",
                     "CER down by about as much as H1 (>= 0.03)", "add")
    s.edit("add", [B], hypothesis=h, delta=delta)
    m2 = s.measure("H2 remove mean offset", rule_mask=MASK)
    s.keep("constant offset removed")
else:
    # H4: tied embedding mild pull toward the donor (expected: no gain)
    E = "dec.embed.tokens"
    h = s.hypothesis("tied embedding may be slightly damaged too; a 15 % pull toward the donor",
                     "CER change within +-0.005; not kept", "interpolate")
    s.edit("interpolate", [E], hypothesis=h, per_tensor={E: {"other": P[E]}}, alpha=0.15)
    m4 = s.measure("H4 embed pull 0.15", rule_mask=MASK)
    s.keep("embedding pulled 0.15 toward donor")

    # H3: slight scale of the restored bias (expected: no gain)
    h = s.hypothesis("restored bias might be slightly too strong for this hand; scale 0.9",
                     "CER change under 0.005; not kept", "scale")
    s.edit("scale", [B], hypothesis=h, factor=0.9)
    m3 = s.measure("H3 bias x0.9", rule_mask=MASK)
    s.keep("bias scaled 0.9")

summary = s.freeze()
