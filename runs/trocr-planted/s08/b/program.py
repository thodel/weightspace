from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
names = [f"dec.08.{blk}.{m}.weight" for blk in ("self", "cross") for m in ("q", "k", "v", "o")]
names += ["dec.08.ffn.in.weight", "dec.08.ffn.out.weight"]
h = s.hypothesis("dec.08 weight matrices carry noise (distance 0.38-0.77, norm ratio up to 1.35); pull 85% toward donor",
                 "CER down by several points", "interpolate")
s.edit("interpolate", names, hypothesis=h, alpha=0.85,
       per_tensor={n: {"other": P[n]} for n in names})
s.keep_unmeasured("dec.08 weights far off the donor in distance and norm, rest healthy")
s.freeze()
