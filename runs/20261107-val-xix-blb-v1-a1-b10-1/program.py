import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
B = "O_18.lin.bias"
nb = T[B].shape[0]


def bias_delta(spec):
    d = np.zeros(nb, dtype=T[B].dtype)
    for ch, v in spec.items():
        if ch in cm:
            d[cm[ch]] += v
    return d


s.measure("C0 baseline")


# H1 blank bias -0.5
d = np.zeros(nb, dtype=T[B].dtype); d[0] = -0.5
h = s.hypothesis("blank over-predicted: deletions dominate C0 errors", "CER -0.005..-0.015", "output bias")
s.edit("add", [B], hypothesis=h, delta=d)
s.measure("H1 blank bias -0.5")
k1 = s.keep("H1 blank bias -0.5")

# H2 further step: -0.5 more if H1 kept, else try a smaller -0.2
step = -0.5 if k1 else -0.2
d = np.zeros(nb, dtype=T[B].dtype); d[0] = step
h = s.hypothesis("further blank bias shift", "CER -0.002..-0.006 if H1 kept, else small", "output bias")
s.edit("add", [B], hypothesis=h, delta=d)
s.measure("H2 blank bias %+.1f" % step)
s.keep("H2 blank bias step")

# H3 space bias +1.0
d = np.zeros(nb, dtype=T[B].dtype); d[cm[" "]] = 1.0
h = s.hypothesis("space bias -7.85 too low; space->nothing is the top confusion", "CER -0.003..-0.010", "output bias")
s.edit("add", [B], hypothesis=h, delta=d)
s.measure("H3 space bias +1.0")
k3 = s.keep("H3 space bias +1.0")
if not k3:
    # smaller, since insertions of space also exist
    d = np.zeros(nb, dtype=T[B].dtype); d[cm[" "]] = 0.4
    h = s.hypothesis("space +0.4 (milder, 30 spurious spaces exist)", "CER -0.001..-0.003", "output bias")
    s.edit("add", [B], hypothesis=h, delta=d)
    s.measure("H3b space bias +0.4")
    s.keep("H3b space bias +0.4")

# H4 weakest letter rows +0.4
spec = {c: 0.4 for c in "einrtsa"}
h = s.hypothesis("weak rows of frequent letters are out-competed by blank", "CER -0.002..-0.006", "output bias")
s.edit("add", [B], hypothesis=h, delta=bias_delta(spec))
s.measure("H4 letter biases +0.4")
s.keep("H4 letter biases +0.4")

# H5 blank row scale 0.9
W = "O_18.lin.weight"
w = T[W].copy()
w[0] *= 0.9
h = s.hypothesis("blank row weak; shrinking lowers blank confidence", "CER +/-0.003", "output row scale")
s.edit("set_rows", [W], hypothesis=h, index=[0], values=w[0:1])
s.measure("H5 blank row x0.9")
s.keep("H5 blank row x0.9")

# H6 parent interpolation on LSTM tensors, alpha 0.15 (expected to fail)
donor = KrakenSubject("/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel")
DT = donor.tensors()
names = [n for n in s.subject.tensors() if n.startswith("L_")
         and n in DT and "bias_ih" not in n]
h = s.hypothesis("parent as regulariser toward less BLB overfitting", "CER +0.00..+0.02 (expected rise)", "interpolate donor")
s.edit("interpolate", names, hypothesis=h, per_tensor={n: {"other": DT[n]} for n in names}, alpha=0.15)
s.measure("H6 parent alpha 0.15", rule_mask=s.unseen_by("kraken-german-xix-v1"))
s.keep("H6 parent alpha 0.15")

print(s.freeze())
