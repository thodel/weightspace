import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
s = open_session(".")
base = s.measure("C0")
cm = s.subject.codec_map()
e, n = cm["e"], cm["n"]
W, B = "O_18.lin.weight", "O_18.lin.bias"


def swap_vals(T, with_bias=True):
    w = T[W][[n, e]].copy()
    b = T[B][[n, e]].copy()
    return w, b


def try_edit(text, pred, cls, names, per_tensor, label):
    h = s.hypothesis(text, pred, cls)
    s.edit("set_rows", names, hypothesis=h, index=[e, n], per_tensor=per_tensor)
    m = s.measure(label)
    ok = m.cer < base.cer and m.collapsed <= base.collapsed + 2 and s.keep(label)
    if not ok and s._last_edit is not None:
        s.revert(label + ": not kept")
    return ok, m


T = s.subject.tensors()
w, b = swap_vals(T)
# H1: swap weight rows and biases of e and n
ok, m = try_edit("e and n output rows exchanged: swap rows and biases",
                 "CER 0.244 -> ~0.10", "set_rows", [W, B],
                 {W: {"values": w}, B: {"values": b}}, "H1 swap rows+bias")
if not ok:
    T = s.subject.tensors()
    # H2: weight rows only
    w = T[W][[n, e]].copy()
    ok, m = try_edit("swap weight rows only", "CER -> ~0.11", "set_rows", [W],
                     {W: {"values": w}}, "H2 swap rows only")
if not ok:
    # H3: parent rows by character
    pm = KrakenSubject(PARENT).codec_map()
    P = KrakenSubject(PARENT).tensors()
    wv = P[W][[pm["e"], pm["n"]]].copy()
    bv = P[B][[pm["e"], pm["n"]]].copy()
    ok, m = try_edit("parent rows for e and n", "CER -> ~0.15", "set_rows", [W, B],
                     {W: {"values": wv}, B: {"values": bv}}, "H3 parent rows")
s.note("final kept: %s" % ok)
print(s.freeze())
