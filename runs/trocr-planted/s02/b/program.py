from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()

name = "dec.embed.norm.bias"
h = s.hypothesis(
    "dec.embed.norm.bias carries a large added shift (norm ratio 6.3, cosine 0.12 to parent); restore the parent's bias",
    "large CER drop (the embedding layer-norm output is offset in every decoder input)",
    "transplant")
s.edit("transplant", [name], hypothesis=h, per_tensor={name: {"donor": P[name]}})
s.keep_unmeasured("only tensor far outside the healthy range; all else normal")
s.freeze()
