import numpy as np
from wsp.run import open_session

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()


def tried(label, hyp, pred, cls, op, names, **kw):
    """One hypothesis, one edit, one counted pass, keep by the harness rule."""
    if s.budget_left < 1:
        return False
    h = s.hypothesis(hyp, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label)
    ok = s.keep(label)
    s.note(f"{label}: cer={m.cer:.4f} collapsed={m.collapsed} kept={ok}")
    return ok


base = s.measure("C0")
s.note(f"C0 cer={base.cer:.4f} collapsed={base.collapsed}")

# H1 blank bias down
blank = 0
d = np.zeros_like(T["O_18.lin.bias"]); d[blank] = -0.6
k1 = tried("H1 blank bias -0.6", "blank bias too high; deletions dominate", "CER -0.003..-0.010",
           "output bias add", "add", ["O_18.lin.bias"], delta=d)

# H2 space bias up
if " " in cm:
    d = np.zeros_like(s.subject.tensors()["O_18.lin.bias"]); d[cm[" "]] = 1.5
    k2 = tried("H2 space bias +1.5", "space bias -7.85 implausibly low; space deletions top error",
               "CER -0.004..-0.012", "output bias add", "add", ["O_18.lin.bias"], delta=d)

# H3 low-rank denoise of L_14 input weights
def lowrank(w, r):
    u, sv, vt = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    return ((u[:, :r] * sv[:r]) @ vt[:r]).astype(np.float32)

names3 = ["L_14.layer.weight_ih_l0", "L_14.layer.weight_ih_l0_reverse"]
T = s.subject.tensors()
k3 = tried("H3 L_14 weight_ih rank120", "additive noise in L_14 input weights; truncated SVD removes it",
           "CER -0.005..-0.03 if noised, else worse", "lstm low-rank denoise", "transplant", names3,
           per_tensor={n: {"donor": lowrank(T[n], 120)} for n in names3})
if not k3 and s.budget_left >= 4:
    T = s.subject.tensors()
    tried("H3b L_14 weight_ih rank250", "milder truncation", "small gain or none", "lstm low-rank denoise",
          "transplant", names3, per_tensor={n: {"donor": lowrank(T[n], 250)} for n in names3})

# H4 clip outliers in L_16.weight_ih_l0
T = s.subject.tensors()
n4 = "L_16.layer.weight_ih_l0"
w = T[n4].astype(np.float64)
med = np.median(w); mad = np.median(np.abs(w - med)) * 1.4826
lim = abs(med) + 10.0 * mad
clipped = np.clip(w, -lim, lim).astype(np.float32)
if np.abs(w).max() > np.abs(clipped).max():
    tried("H4 clip L_16 weight_ih outliers", "spike weights up to 4.7 are damage",
          "CER -0.001..-0.005", "lstm outlier clip", "transplant", [n4], donor=clipped)

s.freeze()
