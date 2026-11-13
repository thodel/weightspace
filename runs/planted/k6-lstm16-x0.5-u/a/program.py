from wsp.run import open_session
import numpy as np

s = open_session(".")
m0 = s.measure("C0")
best = m0.cer
L16 = [f"L_16.layer.{k}{r}" for r in ("", "_reverse")
       for k in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")]
L16W = [n for n in L16 if "weight" in n]


def attempt(text, pred, cls, names, op, label, **kw):
    global best
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label)
    if m.collapsed > m0.collapsed + 5:
        s.revert(label + ": collapse increase")
        return False
    if s.keep(label):
        best = m.cer
        return True
    return False


ok = attempt("L_16 shrunk about half; rescale all x2", "CER -0.05 or more", "scale", L16, "scale", "H1", factor=2.0)
if ok:
    attempt("refine L_16 factor", "-0.005 to -0.01", "scale", L16, "scale", "H1b", factor=1.15)
else:
    ok2 = attempt("L_16 weights only x2", "CER -0.03", "scale", L16W, "scale", "H2", factor=2.0)
    if not ok2:
        attempt("L_16 all x1.5", "CER -0.02", "scale", L16, "scale", "H2b", factor=1.5)

cm = s.subject.codec_map()
T = s.subject.tensors()
d = np.zeros_like(T["O_18.lin.bias"])
d[0] = -0.5
if not attempt("blank bias too high, deletions", "CER -0.01 to -0.02", "add", ["O_18.lin.bias"], "add", "H3", delta=d):
    d[0] = -0.25
    attempt("blank bias smaller shift", "CER -0.005", "add", ["O_18.lin.bias"], "add", "H3b", delta=d)

if " " in cm:
    d = np.zeros_like(s.subject.tensors()["O_18.lin.bias"])
    d[cm[" "]] = 1.5
    attempt("space bias very low", "CER -0.005 to -0.01", "add", ["O_18.lin.bias"], "add", "H4", delta=d)

s.freeze()
