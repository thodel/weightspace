from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
donor = KrakenSubject("/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel")
mask = s.unseen_by("kraken-german-xix-v1")
DT = donor.tensors()

used = 0


def bias_delta(pairs):
    d = np.zeros_like(T["O_18.lin.bias"])
    for row, v in pairs:
        d[row] += v
    return d


def try_edit(label, text, pred, op, names, use_donor=False, **kw):
    """one edit + one pass + keep rule; returns kept"""
    global used
    if used >= 10:
        return False
    h = s.hypothesis(text, pred, op)
    s.edit(op, names, hypothesis=h, **kw)
    s.measure(label, rule_mask=mask if use_donor else None)
    used += 1
    return s.keep(label)


s.measure("C0 baseline")
used = 1

# H1/H2: blank bias
ok = try_edit("blank bias -0.5", "blank over-confident, deletions dominate", "CER -0.005..-0.015", "add",
              ["O_18.lin.bias"], delta=bias_delta([(0, -0.5)]))
if ok:
    try_edit("blank bias -1.0 total", "further step", "CER -0.002..-0.006", "add", ["O_18.lin.bias"],
             delta=bias_delta([(0, -0.5)]))
else:
    try_edit("blank bias +0.5", "opposite sign: collapse/insertion regime", "CER -0.002..-0.005", "add",
             ["O_18.lin.bias"], delta=bias_delta([(0, +0.5)]))

# H3: space bias
if ' ' in cm:
    try_edit("space bias +1.0", "space deletions 78 vs insertions 30", "CER -0.004..-0.010", "add",
             ["O_18.lin.bias"], delta=bias_delta([(cm[' '], 1.0)]))

# H4: task-vector extrapolation away from the parent (all tensors the codec growth did not touch)
body = [n for n in T if not n.startswith("O_18") and "bias_ih" not in n]
per = {n: {"other": DT[n]} for n in body}
ok = try_edit("extrapolate -0.25", "amplify fine-tune direction", "CER -0.005..-0.02", "interpolate", body,
              use_donor=True, per_tensor=per, alpha=-0.25)
if ok:
    try_edit("extrapolate another -0.15", "further amplification", "CER -0.002..-0.01", "interpolate", body,
             use_donor=True, per_tensor=per, alpha=-0.15)
else:
    try_edit("shrink toward parent +0.15", "opposite sign", "CER +0.01 (expected to fail)", "interpolate", body,
             use_donor=True, per_tensor=per, alpha=0.15)

# H5: blank row weight scaled 0.9
W = T["O_18.lin.weight"]
dW = np.zeros_like(W)
dW[0] = -0.1 * W[0]
try_edit("blank row x0.9", "weaker blank evidence, input dependent", "CER -0.002..-0.008", "add",
         ["O_18.lin.weight"], delta=dW)

# H6: frequent deleted letters bias
letters = [cm[c] for c in "einrts" if c in cm]
try_edit("letters bias +0.3", "frequent deleted letters", "CER -0.002..-0.006", "add", ["O_18.lin.bias"],
         delta=bias_delta([(r, 0.3) for r in letters]))

s.note("a2 done; passes used %d" % used)
summary = s.freeze()
