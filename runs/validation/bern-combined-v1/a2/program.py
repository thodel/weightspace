"""a2: bias calibration against over-emitted blank, then sibling interpolation (donor rule applied)."""
import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
BIAS = "O_18.lin.bias"
SIB = "kraken-bern-schlussredaktion-v1"
SIB_PATH = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-schlussredaktion-v1.mlmodel"


def bias_delta(shifts):
    d = np.zeros_like(T[BIAS])
    for ch, v in shifts.items():
        if ch == "<blank>":
            d[0] = v
        elif ch in cm:
            d[cm[ch]] = v
    return d


def try_bias(text, pred, shifts, label):
    h = s.hypothesis(text, pred, "bias shift")
    s.edit("add", [BIAS], hypothesis=h, delta=bias_delta(shifts))
    s.measure(label)
    return s.keep(label)


base = s.measure("C0 baseline")
s.note(f"baseline CER {base.cer:.4f}")

# H1 / H2: blank bias
k1 = try_bias("blank bias too high: deletions dominate", "CER -0.004..-0.012", {"<blank>": -0.3}, "blank -0.3")
if k1:
    try_bias("further blank lowering", "CER -0.002..-0.006", {"<blank>": -0.3}, "blank -0.6 cum")
else:
    s.note("H1 failed: blank bias not the lever; skip H2")

# H3: space
try_bias("space deleted 195 vs inserted 99", "CER -0.002..-0.005", {" ": 0.3}, "space +0.3")

# H4: slash
try_bias("slash deleted 98 vs inserted 57", "CER -0.001..-0.003", {"/": 0.3}, "slash +0.3")

# H5 / H6: sibling interpolation on all tensors but the output layer, judged on lines the sibling did not train on
donor = KrakenSubject(SIB_PATH)
DT = donor.tensors()
names = [n for n in T if not n.startswith("O_18") and n in DT and DT[n].shape == T[n].shape]
mask = s.unseen_by(SIB)
h = s.hypothesis("sibling shares parent; averaging toward it improves unseen lines", "CER -0.003..-0.010", "interpolate (donor)")
s.edit("interpolate", names, hypothesis=h, per_tensor={n: {"other": DT[n]} for n in names}, alpha=0.25)
s.measure("interp 0.25 toward schlussredaktion", rule_mask=mask)
k5 = s.keep("interp 0.25")
if k5:
    # total alpha 0.5: from the current point (0.75*w + 0.25*d), (1-a)*cur + a*d with a=1/3 gives 0.5 overall
    h = s.hypothesis("more of the sibling", "CER -0.002..-0.005", "interpolate (donor)")
    s.edit("interpolate", names, hypothesis=h, per_tensor={n: {"other": DT[n]} for n in names}, alpha=1.0 / 3.0)
    s.measure("interp 0.5 total", rule_mask=mask)
    s.keep("interp 0.5 total")
else:
    s.note("H5 failed: no interpolation toward sibling")

# H7: final blank refinement
try_bias("rows moved; recalibrate blank", "CER -0.001..-0.003", {"<blank>": -0.15}, "blank -0.15 final")

summary = s.freeze()
