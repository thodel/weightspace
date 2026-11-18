import numpy as np
from wsp.run import open_session

s = open_session(".")
MASK = s.unseen_by()
T = s.subject.tensors()

Z = ["dec.05.cross.o.weight", "dec.05.cross.o.bias"]
N4 = ["dec.04.cross.o.weight", "dec.04.cross.o.bias"]
N6 = ["dec.06.cross.o.weight", "dec.06.cross.o.bias"]


def zero_layer_present():
    return all(not np.any(T[n]) for n in Z)


def try_edit(text, pred, cls, op, names, per_tensor, label, reason, **kw):
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kw)
    m = s.measure(label, rule_mask=MASK)
    if m.collapsed > base_collapsed + 5:
        s.revert("collapsed lines rose: %d" % m.collapsed)
        return False, m
    ok = s.keep(reason)
    return ok, m


m0 = s.measure("C0", rule_mask=MASK)
base_collapsed = m0.collapsed
s.note("C0 cer %.4f collapsed %d" % (m0.cer, m0.collapsed))

repaired = False
if zero_layer_present() and s.budget_left >= 1:
    # H1: average of neighbours
    pt = {n: {"donor": 0.5 * (T[a] + T[b])} for n, a, b in zip(Z, N4, N6)}
    repaired, m1 = try_edit(
        "dec.05 cross.o is all zero; rebuild as mean of dec.04 and dec.06 cross.o",
        "CER -0.01 to -0.04", "transplant", "transplant", Z, pt,
        "H1 mean-neighbours", "H1 neighbour mean for zeroed dec.05.cross.o")
    if repaired and s.budget_left >= 1:
        T = s.subject.tensors()
        # H1b: strengthen
        h = s.hypothesis("rebuilt dec.05 cross.o is too weak at 0.5 average; scale x1.6",
                         "CER -0.003 to -0.01", "scale")
        s.edit("scale", Z, hypothesis=h, factor=1.6)
        m = s.measure("H1b x1.6", rule_mask=MASK)
        s.keep("H1b scale up")
    if not repaired:
        T = s.subject.tensors()
        for tag, src in (("dec.04", N4), ("dec.06", N6)):
            if s.budget_left < 3:
                break
            pt = {n: {"donor": 0.6 * T[a]} for n, a in zip(Z, src)}
            # T may be a live view; the zeroed layer is unchanged after revert
            ok, _ = try_edit(
                "dec.05 cross.o from %s only, scaled 0.6" % tag,
                "CER -0.005 to -0.02", "transplant", "transplant", Z, pt,
                "H2 from " + tag, "H2 single neighbour " + tag)
            T = s.subject.tensors()
            if ok:
                repaired = True
                break

if s.budget_left >= 1:
    T = s.subject.tensors()
    h = s.hypothesis("dec.11.norm_ffn.weight norm 67 vs ~31: maybe amplified; scale x0.8",
                     "CER 0 +- 0.005, likely reverted", "scale")
    s.edit("scale", ["dec.11.norm_ffn.weight"], hypothesis=h, factor=0.8)
    m = s.measure("H5 dec.11 norm_ffn x0.8", rule_mask=MASK)
    s.keep("H5 last-layer norm")

summary = s.freeze()
