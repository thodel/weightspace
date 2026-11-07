from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
DONOR = "kraken-german-xix-v1"
donor = KrakenSubject("/home/tobias/Repo/weightspace/models/kraken/%s.mlmodel" % DONOR)
nrows = T["O_18.lin.bias"].shape[0]

def bias_delta(vals):
    d = np.zeros(nrows, dtype=np.float32)
    for r, v in vals.items():
        if r is not None:
            d[r] += v
    return d

s.measure("C0 baseline")

# H1: blank bias down
h = s.hypothesis("child over-predicts blank on unseen hands, causing deletions",
                 "CER -0.005 to -0.015", "add (O_18 bias, blank)")
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta({0: -0.4}))
s.measure("H1 blank bias -0.4")
if s.keep("blank bias -0.4"):
    h = s.hypothesis("further blank reduction helps", "CER -0.002 to -0.008", "add (O_18 bias, blank)")
    s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta({0: -0.4}))
    s.measure("H1b blank bias -0.4 more")
    blank_kept = True
    if not s.keep("blank bias -0.8 total"):
        pass
else:
    blank_kept = False
    h = s.hypothesis("a smaller blank reduction may still help", "CER -0.001 to -0.004", "add (O_18 bias, blank)")
    s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta({0: -0.2}))
    s.measure("H1c blank bias -0.2")
    blank_kept = s.keep("blank bias -0.2")

# H2: space bias up
h = s.hypothesis("space deleted 78x with bias -7.85; raising recovers some", "CER -0.002 to -0.006", "add (O_18 bias, space)")
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta({cm.get(" "): 0.5}))
s.measure("H2 space bias +0.5")
s.keep("space bias +0.5")

# H3: frequent letters bias up
h = s.hypothesis("frequent letters weakest and most deleted", "CER -0.002 to -0.008", "add (O_18 bias, letters)")
s.edit("add", ["O_18.lin.bias"], hypothesis=h,
       delta=bias_delta({cm.get(c): 0.3 for c in "einrtsadh"}))
s.measure("H3 letters bias +0.3")
s.keep("letters bias +0.3")

# H4: extrapolate away from parent on shared-shape tensors
names = [n for n in T if not n.startswith("O_18")]
dT = donor.tensors()
h = s.hypothesis("extend the BLB adaptation direction beyond the child", "CER -0.003 to -0.010", "interpolate alpha=-0.1 (donor parent)")
s.edit("interpolate", names, hypothesis=h, per_tensor={n: {"other": dT[n]} for n in names}, alpha=-0.1)
s.measure("H4 extrapolate -0.1", rule_mask=s.unseen_by(DONOR))
if s.keep("extrapolation -0.1"):
    h = s.hypothesis("more extrapolation", "CER -0.001 to -0.005", "interpolate alpha=-0.1 more (donor parent)")
    s.edit("interpolate", names, hypothesis=h, per_tensor={n: {"other": dT[n]} for n in names}, alpha=-0.1)
    s.measure("H4b extrapolate another -0.1", rule_mask=s.unseen_by(DONOR))
    s.keep("extrapolation -0.2 total")

# H5: shrink blank row weights, only if the blank bias edit was kept
if blank_kept:
    h = s.hypothesis("blank row logit shape also dominant", "CER -0.001 to -0.004", "scale (O_18 row 0)")
    w = T["O_18.lin.weight"]
    d = np.zeros_like(w)
    d[0] = -0.05 * w[0]
    s.edit("add", ["O_18.lin.weight"], hypothesis=h, delta=d)
    s.measure("H5 blank row x0.95")
    s.keep("blank row x0.95")

summary = s.freeze()
print(summary)
