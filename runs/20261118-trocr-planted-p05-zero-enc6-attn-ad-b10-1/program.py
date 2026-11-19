from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

O = ["enc.06.attn.o.weight", "enc.06.attn.o.bias"]
donor = {n: {"donor": P[n]} for n in O}
other = {n: {"other": P[n]} for n in O}

base = s.measure("C0", rule_mask=MASK)
s.note("C0 cer %.4f collapsed %d" % (base.cer, base.collapsed))
cur = base.cer

# H1: restore the zeroed enc.06 attention output projection from the parent
h = s.hypothesis("enc.06.attn.o is exactly zero; restoring parent values returns the layer-6 attention path",
                 "CER -8 to -18 points, fewer collapsed lines", "transplant")
s.edit("transplant", O, hypothesis=h, per_tensor=donor)
m = s.measure("H1 transplant enc.06 attn.o", rule_mask=MASK)
kept1 = s.keep("enc.06.attn.o restored from parent")
if kept1:
    cur = m.cer

if kept1:
    # H2: gain of the repaired projection
    h = s.hypothesis("donor o may need a different gain for this child", "within +-0.5 points", "scale")
    s.edit("scale", O, hypothesis=h, factor=1.15)
    m = s.measure("H2 scale 1.15", rule_mask=MASK)
    if not s.keep("gain 1.15 of repaired o"):
        pass
    else:
        cur = m.cer
else:
    # H3: gentler repair, halfway to the parent
    h = s.hypothesis("donor o partly compatible; half repair", "-1 to -6 points", "interpolate")
    s.edit("interpolate", O, hypothesis=h, alpha=0.5, per_tensor=other)
    m = s.measure("H3 interpolate 0.5", rule_mask=MASK)
    if s.keep("half repair of enc.06 attn.o"):
        cur = m.cer

if cur > 0.20:
    # H4: tied embedding toward the parent
    h = s.hypothesis("embedding drift is residual damage", "+-1 point, expect revert", "interpolate")
    s.edit("interpolate", ["dec.embed.tokens"], hypothesis=h, alpha=0.2,
           per_tensor={"dec.embed.tokens": {"other": P["dec.embed.tokens"]}})
    m = s.measure("H4 embed 0.2 to parent", rule_mask=MASK)
    if s.keep("embedding toward parent"):
        cur = m.cer

s.note("final cer %.4f" % cur)
summary = s.freeze()
