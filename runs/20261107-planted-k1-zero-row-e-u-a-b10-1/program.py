import numpy as np
from wsp.run import open_session

s = open_session(".")
m0 = s.measure("C0")
cm = s.subject.codec_map()
E = cm["e"]
W = lambda: s.subject.tensors()["O_18.lin.weight"]
used = 1
BUDGET = 10


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def build(w, with_blank=False):
    conf = {"a": 131, "i": 90, "o": 52, "r": 36, "c": 35, "u": 32}
    conf = {c: n for c, n in conf.items() if c in cm}
    tot = float(sum(conf.values()))
    mc = sum(w[cm[c]] * (n / tot) for c, n in conf.items())
    others = [r for r in range(w.shape[0]) if r not in (0, E)]
    d = mc - w[others].mean(axis=0)
    d = unit(d)
    if with_blank:
        d = unit(d + 0.5 * unit(w[0]))
    return d


def try_row(label, row, text, pred):
    global used
    if used >= BUDGET:
        return False
    h = s.hypothesis(text, pred, "set_rows")
    s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h, index=[E], values=row[None, :].astype(np.float32))
    m = s.measure(label)
    used += 1
    return s.keep(label)


w0 = W().copy()
best_dir = build(w0)
norm = 6.0
kept = try_row("H1 e row from confusers, norm 6", norm * best_dir,
               "e row rebuilt from a,i,o,r,c,u minus mean", "CER 0.162 -> 0.10-0.13")

if kept:
    d2 = build(w0, with_blank=True)
    if try_row("H2 + blank component", norm * d2, "add blank direction to e row", "gain 0.005-0.015 or loss"):
        best_dir = d2
    for _ in range(2):
        nn = norm * 1.5
        if try_row("H3 norm x1.5", nn * best_dir, "louder e row", "gain 0.005-0.02"):
            norm = nn
        else:
            break
    for sign in (1.0, -1.0):
        if used >= BUDGET:
            break
        d = np.zeros(297, dtype=np.float32)
        d[E] = sign
        h = s.hypothesis("e bias shift %+.0f" % sign, "|dCER| <= 0.01", "add")
        s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=d)
        s.measure("H4 bias %+.0f" % sign)
        used += 1
        if s.keep("bias %+.0f" % sign):
            break
else:
    fb = np.mean([w0[cm[c]] for c in "aou" if c in cm], axis=0) * 0.6
    try_row("H5 fallback e = 0.6 mean(a,o,u)", fb, "e row from a,o,u rows", "gain 0.01-0.03")

summary = s.freeze()
