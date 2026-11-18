from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()

FFN = ["enc.03.ffn.out.weight", "enc.03.ffn.out.bias"]


def was_kept(k):
    if isinstance(k, bool):
        return k
    for a in ("kept", "accepted"):
        if hasattr(k, a):
            return bool(getattr(k, a))
    if isinstance(k, dict):
        return bool(k.get("kept", False))
    return bool(k)


m0 = s.measure("C0", rule_mask=MASK)
s.note("C0 cer=%.4f collapsed=%s" % (m0.cer, m0.collapsed))
best = m0.cer
fixed = False

# H1: enc.03 FFN output was halved
h = s.hypothesis("enc.03 ffn.out weight+bias were multiplied by ~0.5 (norm ratio 0.5000 vs donor); restore with x2",
                 "CER down by 0.04 to 0.10, fewer collapsed lines", "scale")
s.edit("scale", FFN, hypothesis=h, factor=2.0)
m1 = s.measure("H1 enc03 ffn out x2", rule_mask=MASK)
if was_kept(s.keep("H1 restore halved enc.03 ffn.out")):
    fixed = True
    best = m1.cer
else:
    # H2: smaller factors
    for f in (1.8, 1.5):
        h = s.hypothesis("x2 not accepted; try factor %.1f on enc.03 ffn.out" % f,
                         "CER down 0.02 to 0.08 if the tensor is still too small", "scale")
        s.edit("scale", FFN, hypothesis=h, factor=f)
        m = s.measure("H2 enc03 ffn out x%.1f" % f, rule_mask=MASK)
        if was_kept(s.keep("H2 factor %.1f" % f)):
            fixed = True
            best = m.cer
            break

if fixed:
    # H3: pull the repaired tensor slightly toward the donor
    h = s.hypothesis("enc.03 ffn.out interpolated 0.25 toward donor", "change about 0 (+-0.003); expect revert",
                     "interpolate")
    s.edit("interpolate", FFN, hypothesis=h, alpha=0.25,
           per_tensor={n: {"other": P[n]} for n in FFN})
    m3 = s.measure("H3 enc03 ffn out 25% donor", rule_mask=MASK)
    if not was_kept(s.keep("H3 donor pull")):
        pass
    else:
        best = m3.cer

# H4: embedding probe (low prior)
h = s.hypothesis("tied embedding drift partly harmful: interpolate dec.embed.tokens 0.15 toward donor",
                 "change +0.00 to +0.01; expect revert", "interpolate")
s.edit("interpolate", ["dec.embed.tokens"], hypothesis=h, alpha=0.15,
       per_tensor={"dec.embed.tokens": {"other": P["dec.embed.tokens"]}})
m4 = s.measure("H4 embedding 15% donor", rule_mask=MASK)
s.keep("H4 embedding donor pull")

s.note("final best cer (before H4 decision) %.4f" % best)
summary = s.freeze()
print(summary)
