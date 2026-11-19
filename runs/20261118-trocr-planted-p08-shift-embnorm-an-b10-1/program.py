from wsp.run import open_session
import numpy as np

s = open_session(".")
MASK = s.unseen_by()
base = s.measure("C0", rule_mask=MASK)
best = base


def attempt(label, text, pred, cls, op, names, **kw):
    global best
    if s.budget_left < 1:
        return None
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=MASK)
    bad = m.collapsed > best.collapsed + 5
    if (not bad) and m.cer < best.cer and s.keep(label):
        best = m
        return True
    if (not bad) and m.cer >= best.cer:
        s.revert(label + ": no gain")
    elif bad:
        s.revert(label + ": more collapsed lines")
    return False


# H1: last decoder FFN norm gain inflated
n1 = ["dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"]
c0 = best.cer
ok = attempt("H1 dec11 norm_ffn x0.5", "dec.11.norm_ffn gain inflated ~x2.2",
             "CER -0.02..-0.08", "scale", "scale", n1, factor=0.5)
if ok is False and abs(best.cer - c0) < 0.01 and s.budget_left >= 6:
    attempt("H1b dec11 norm_ffn x0.75", "milder inflation", "CER -0.01..-0.03",
            "scale", "scale", n1, factor=0.75)

# H2: encoder layer 10 FFN norm
attempt("H2 enc10 norm_ffn x0.5", "enc.10.norm_ffn gain inflated",
        "CER -0.01..-0.05", "scale", "scale",
        ["enc.10.norm_ffn.weight", "enc.10.norm_ffn.bias"], factor=0.5)

# H3: cross-attention output damped in all decoder layers
names = []
for i in range(12):
    names += [f"dec.{i:02d}.cross.o.weight", f"dec.{i:02d}.cross.o.bias"]
attempt("H3 cross.o x1.15", "cross-attention damped, causing deletions",
        "CER -0.005..-0.02", "scale", "scale", names, factor=1.15)

# H4: late decoder FFN output weak
names = []
for i in range(8, 12):
    names += [f"dec.{i:02d}.ffn.out.weight", f"dec.{i:02d}.ffn.out.bias"]
attempt("H4 dec08-11 ffn.out x1.1", "late decoder FFN weak",
        "CER -0.005..-0.015", "scale", "scale", names, factor=1.1)

s.note("final CER %.4f (C0 %.4f)" % (best.cer, base.cer))
s.freeze()
