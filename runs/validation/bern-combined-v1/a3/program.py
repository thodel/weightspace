import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"
s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()

m0 = s.measure("C0 baseline")


def bias_delta(shifts):
    d = np.zeros_like(T["O_18.lin.bias"])
    for ch, v in shifts.items():
        row = 0 if ch == "<blank>" else cm.get(ch)
        if row is not None:
            d[row] += v
    return d


def bias_step(label, shifts, text, pred):
    h = s.hypothesis(text, pred, "additive output bias")
    s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=bias_delta(shifts))
    s.measure(label)
    return s.keep(label)


def donor_arrays(donor, alpha_names, rows):
    """Arrays aligned to the subject: same-shaped tensors direct, O_18 rows by character."""
    dT = donor.tensors()
    dcm = donor.codec_map()
    out = {}
    for n in alpha_names:
        out[n] = dT[n].copy()
    if rows:
        for n in ("O_18.lin.weight", "O_18.lin.bias"):
            a = T[n].copy()
            a[0] = dT[n][0]
            for ch, r in cm.items():
                if ch in dcm:
                    a[r] = dT[n][dcm[ch]]
            out[n] = a
    return out


def merge_step(donor_id, label, alpha, rows, text, pred):
    donor = KrakenSubject(MODELS + donor_id + ".mlmodel")
    if rows:
        names = ["O_18.lin.weight", "O_18.lin.bias"]
    else:
        names = [n for n in T if not n.startswith("O_18") and n in donor.tensors()
                 and donor.tensors()[n].shape == T[n].shape]
    arr = donor_arrays(donor, names, rows)
    h = s.hypothesis(text, pred, "interpolate toward donor")
    s.edit("interpolate", names, hypothesis=h, alpha=alpha,
           per_tensor={n: {"other": arr[n]} for n in names})
    s.measure(label, rule_mask=s.unseen_by(donor_id))
    return s.keep(label)


SR = "kraken-bern-schlussredaktion-v1"
HV = "kraken-bern-hertwig-v2"

# H1: blank bias down
k1 = bias_step("blank bias -0.3", {"<blank>": -0.3},
               "Blank is over-predicted (deletions dominate, length ratio 0.949); lowering its bias restores characters.",
               "CER -0.005 to -0.015")
if k1:
    bias_step("blank bias -0.3 more", {"<blank>": -0.3},
              "Further blank lowering continues to pay while deletions exceed insertions.",
              "additional CER -0.002 to -0.007, else optimum near -0.3")

# H2: space bias up
bias_step("space bias +0.3", {" ": 0.3},
          "Space is the top deletion (195 vs 99 inserted); raising its prior recovers spaces.",
          "CER -0.003 to -0.008")

# H3: slash bias up
bias_step("slash bias +0.3", {"/": 0.3},
          "Slash deleted 98 times, inserted 57; small net deletion excess.",
          "CER -0.001 to -0.004")

# H4: sibling interpolation, non-output tensors
k4 = merge_step(SR, "interp SR alpha .25 body", 0.25, False,
                "Fine-tuned siblings of one parent are linearly connected; SR is better on all lines (0.118 vs 0.139).",
                "CER -0.005 to -0.02 on SR-unseen lines")
if k4:
    # H5: output rows
    merge_step(SR, "interp SR alpha .25 output rows", 0.25, True,
               "Output rows aligned by character move toward the sibling's with the body.",
               "additional CER -0.001 to -0.005")
    # H6: further body step: from x*(0.75)+0.25*SR to 0.5 total needs alpha 1/3 of the remainder
    merge_step(SR, "interp SR body to 0.5 total", 1.0 / 3.0, False,
               "If the first step helped, the optimum may lie further toward the sibling.",
               "additional CER -0.002 to -0.008, else revert")
    bias_step("post-merge blank bias -0.2", {"<blank>": -0.2},
              "The merge shifts blank calibration; a small re-tune.", "CER -0.003 or less")
else:
    # H7: parent is worse; one cheap probe
    merge_step(HV, "interp HV alpha .25 body", 0.25, False,
               "Parent interpolation is unlikely to help (CER 0.210).",
               "CER -0.002 to +0.01, expected refused")

summary = s.freeze()
