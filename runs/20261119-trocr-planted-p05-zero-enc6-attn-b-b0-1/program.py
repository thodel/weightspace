from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
names = ["enc.06.attn.o.weight", "enc.06.attn.o.bias"]
h = s.hypothesis(
    "enc.06 attention output projection (weight and bias) is zeroed (norm ratio 0, distance 2.0 vs parent)",
    "large CER drop (many deletions: spaces, e, n, r); restores encoder layer 6 attention path",
    "transplant")
s.edit("transplant", names, hypothesis=h, per_tensor={n: {"donor": P[n]} for n in names})
s.keep_unmeasured("enc.06.attn.o zeroed; restored from donor, all else within healthy range")
s.freeze()
