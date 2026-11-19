from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
names = ["dec.05.cross.o.weight", "dec.05.cross.o.bias"]
h = s.hypothesis(
    "dec.05.cross.o (weight and bias) is exactly zero in the subject (norm ratio 0, distance 2.0 to donor): "
    "cross-attention of decoder layer 5 is dead; restore from the donor",
    "CER falls clearly (several points; spaces and letters dropped in C0 come back)",
    "transplant")
s.edit("transplant", names, hypothesis=h, per_tensor={n: {"donor": P[n]} for n in names})
s.keep_unmeasured("only tensor off the healthy-child range; zeroed output projection of a cross-attention")
s.freeze()
