from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
T = s.subject.tensors()
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()

name = "dec.embed.tokens"
zero = [i for i in range(T[name].shape[0]) if not np.any(T[name][i])]
if zero:
    h = s.hypothesis("rows of the tied embedding that are exactly zero are restored from the donor",
                     "CER down several points; deleted , . t e en und come back", "set_rows")
    s.edit("set_rows", [name], hypothesis=h, index=zero,
           values=np.asarray(P[name][zero], dtype=np.float32))
    s.keep_unmeasured("six zeroed token rows restored from donor")
s.freeze()
