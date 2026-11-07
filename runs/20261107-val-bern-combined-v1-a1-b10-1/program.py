import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
SCH = "kraken-bern-schlussredaktion-v1"
BL = 0
mask = None  # becomes unseen_by(SCH) once a donor edit is kept


def bias_delta(pairs):
    d = np.zeros_like(T["O_18.lin.bias"])
    for row, v in pairs:
        if row is not None:
            d[row] = v
    return d


m0 = s.measure("C0 baseline")

# H1 blank bias down
h = s.hypothesis("blank bias too high: model under-emits", "CER -0.003..-0.010", "add on O_18 bias")
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta([(BL, -0.3)]))
s.measure("blank bias -0.3", rule_mask=mask)
k1 = s.keep("blank bias -0.3")

# H2
step = -0.3 if k1 else -0.15
h = s.hypothesis("second blank-bias step" if k1 else "smaller blank-bias step",
                 "further -0.001..-0.004" if k1 else "-0.001..-0.003", "add on O_18 bias")
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta([(BL, step)]))
s.measure("blank bias step 2", rule_mask=mask)
s.keep("blank bias step 2")

# H3 blank row scale
h = s.hypothesis("blank row weaker", "0..-0.003", "scale blank row")
W = T["O_18.lin.weight"]
d = np.zeros_like(W)
d[BL] = -0.1 * W[BL]
s.edit("add", ["O_18.lin.weight"], hypothesis=h, delta=d)
s.measure("blank row x0.9", rule_mask=mask)
s.keep("blank row x0.9")

# H4 donor interpolation, hidden tensors only
donor = KrakenSubject("/home/tobias/Repo/weightspace/models/kraken/%s.mlmodel" % SCH)
DT = donor.tensors()
hid = [n for n in T if not n.startswith("O_18") and n in DT and DT[n].shape == T[n].shape]
orig = {n: T[n].copy() for n in hid}
dmask = s.unseen_by(SCH)
h = s.hypothesis("sibling schlussredaktion shares ancestor; interpolate hidden tensors alpha .25",
                 "-0.002..-0.008 on donor-unseen lines", "interpolate")
s.edit("interpolate", hid, hypothesis=h, per_tensor={n: {"other": DT[n]} for n in hid}, alpha=0.25)
s.measure("interp sch 0.25", rule_mask=dmask)
if s.keep("interpolate toward sch 0.25"):
    mask = dmask
    # H5 total alpha 0.5 relative to the original hidden tensors
    h = s.hypothesis("extrapolate alpha to 0.5", "-0.001..-0.005 or rise", "interpolate")
    s.edit("interpolate", hid, hypothesis=h, per_tensor={n: {"other": DT[n]} for n in hid}, alpha=1.0 / 3.0)
    s.measure("interp sch 0.5", rule_mask=mask)
    s.keep("interpolate toward sch 0.5")

# H6 space bias up
h = s.hypothesis("spaces deleted twice as often as inserted", "-0.002..-0.006", "add on O_18 bias")
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta([(cm.get(" "), 0.2)]))
s.measure("space bias +0.2", rule_mask=mask)
s.keep("space bias +0.2")

# H7 slash bias up
h = s.hypothesis("'/' deleted more than inserted", "-0.001..-0.003", "add on O_18 bias")
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta([(cm.get("/"), 0.2)]))
s.measure("slash bias +0.2", rule_mask=mask)
s.keep("slash bias +0.2")

# H8 frequent letters
h = s.hypothesis("frequent letters deleted", "-0.001..-0.004", "add on O_18 bias")
s.edit("add", ["O_18.lin.bias"], hypothesis=h,
       delta=bias_delta([(cm.get(c), 0.1) for c in "entris"]))
s.measure("letter bias +0.1", rule_mask=mask)
s.keep("letter bias +0.1")

s.freeze()
