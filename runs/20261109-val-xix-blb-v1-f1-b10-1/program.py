"""Programmer f1 (claude-fable-5-1) for kraken-german-xix-blb-v1. Written in advance from the pack;
see preregistration_hypotheses.md. Budget 10 passes: pass 1 = C0, then 9 measured edits, each decided by s.keep.

Reading of the pack: length ratio 0.970 and seven deletions among the ten top confusions -> the model
under-emits; the blank row is the weakest row with a positive bias, the deleted letters are the weakest
letter rows, the space bias is the most negative of all. The parent is 2.4x worse on these lines (weak donor).
"""
from wsp.run import open_session
from wsp.align import common_names
import numpy as np

DONOR_ID = "kraken-german-xix-v1"
DONOR_PATH = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
BIAS, WEIGHT = "O_18.lin.bias", "O_18.lin.weight"
n_rows = T[BIAS].shape[0]
BLANK = 0
SPACE = cm.get(" ")
s.note(f"f1 start: {n_rows} output rows, blank bias {T[BIAS][BLANK]:+.3f}, "
       f"space row {SPACE} bias {T[BIAS][SPACE]:+.3f}" if SPACE is not None else "f1 start: no single-label space row")


def report(label, m):
    print(f"{label}: CER {m.cer:.4f} collapsed {m.collapsed} passes left {s.budget_left}")


def attempt(h, op, names, label, rule_mask=None, **kw):
    """One edit, one counted pass, one keep decision. Returns True/False, or None when no pass is left."""
    if s.budget_left < 1:
        s.note(f"skipped {label}: no passes left")
        return None
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=rule_mask)
    kept = s.keep(label)
    report(f"{label} ({'kept' if kept else 'refused'})", m)
    return bool(kept)


def bias_delta(row, amount):
    d = np.zeros(n_rows, dtype=np.float32)
    d[row] = amount
    return d


# ---- pass 1: C0 baseline -------------------------------------------------------------------------------
m0 = s.measure("C0 baseline")
report("C0", m0)

# ---- H1 / H1b: blank bias (the under-emission lever) ---------------------------------------------------
blank_total = 0.0
blank_steps_kept = []
h = s.hypothesis("H1: blank bias -0.5; deletions (space 78, e 44, i 29, n 26, r 23, t 21, s 21) outnumber insertions, "
                 "length ratio 0.970", "CER -0.005 (-0.002 .. -0.010)", "output bias shift")
k1 = attempt(h, "add", [BIAS], "H1 blank bias -0.5", delta=bias_delta(BLANK, -0.5))
if k1:
    blank_total -= 0.5; blank_steps_kept.append(-0.5)
    h = s.hypothesis("H1b: blank bias a further -0.5 (dose step after a kept -0.5)", "CER -0.002 (-0.004 .. +0.002)",
                     "output bias shift")
    k1b = attempt(h, "add", [BIAS], "H1b blank bias -0.5 more", delta=bias_delta(BLANK, -0.5))
    if k1b:
        blank_total -= 0.5; blank_steps_kept.append(-0.5)
else:
    h = s.hypothesis("H1b: blank bias -0.25 (smaller dose after a refused -0.5)", "CER -0.002 (-0.001 .. -0.004)",
                     "output bias shift")
    k1b = attempt(h, "add", [BIAS], "H1b blank bias -0.25", delta=bias_delta(BLANK, -0.25))
    if k1b:
        blank_total -= 0.25; blank_steps_kept.append(-0.25)

# ---- H2 / H2b: space bias ------------------------------------------------------------------------------
if SPACE is not None:
    h = s.hypothesis("H2: space bias +0.6; 78 deleted vs 30 inserted spaces, space bias is the most negative row (-7.85)",
                     "CER -0.002 (-0.001 .. -0.005)", "output bias shift")
    k2 = attempt(h, "add", [BIAS], "H2 space bias +0.6", delta=bias_delta(SPACE, 0.6))
    if k2:
        h = s.hypothesis("H2b: space bias a further +0.6 (dose step after a kept +0.6)", "CER -0.001 (-0.003 .. +0.002)",
                         "output bias shift")
        attempt(h, "add", [BIAS], "H2b space bias +0.6 more", delta=bias_delta(SPACE, 0.6))
    else:
        h = s.hypothesis("H2b: space bias +0.3 (smaller dose after a refused +0.6)", "CER -0.001 (-0.003 .. +0.001)",
                         "output bias shift")
        attempt(h, "add", [BIAS], "H2b space bias +0.3", delta=bias_delta(SPACE, 0.3))
else:
    s.note("H2/H2b skipped: no single-label space row in the codec")

# ---- H3: scale the weight rows of the weakest frequent letters by 1.08 ---------------------------------
weak = [cm[c] for c in "erstina" if c in cm]
if weak:
    W = T[WEIGHT]
    d = np.zeros_like(W, dtype=np.float32)
    d[weak] = 0.08 * W[weak]
    h = s.hypothesis(f"H3: weight rows of e r s t i n a (rows {weak}) scaled by 1.08; the deleted letters are the weakest "
                     "letter rows (7.6-8.3 vs 14.8 mean); evidence-conditioned boost unlike a bias",
                     "CER -0.002 (-0.001 .. -0.003)", "output row scaling")
    attempt(h, "add", [WEIGHT], "H3 weak letter rows x1.08", delta=d)
else:
    s.note("H3 skipped: none of e r s t i n a is a single-label row")

# ---- H4 / H4b: interpolation toward the parent (weak donor; donor rule) --------------------------------
donor = None
try:
    from wsp.kraken_subject import KrakenSubject
    donor = KrakenSubject(DONOR_PATH)
except Exception as e:  # the donor file may be unreadable in the sandbox; the hypotheses then stay untested
    s.note(f"H4/H4b skipped: donor could not be loaded ({e})")

if donor is not None:
    mask = s.unseen_by(DONOR_ID)
    n_rule = sum(1 for x in mask if x)
    if n_rule == 0:
        s.note("H4/H4b skipped: donor rule leaves no practice line to judge on")
    else:
        DT = donor.tensors()
        shared = [n for n in common_names(T, DT) if not n.startswith("O_18")]
        s.note(f"donor {DONOR_ID}: {len(shared)} shared conv/LSTM tensors, rule lines {n_rule}/{len(mask)}")
        h = s.hypothesis(f"H4: interpolate all {len(shared)} conv+LSTM tensors toward the parent at alpha 0.1 "
                         "(WiSE-FT-style regularisation; parent CER 0.3266 on these lines, so a refusal is expected)",
                         "CER +0.002 (0.000 .. +0.005)", "interpolation (donor: parent)")
        k4 = attempt(h, "interpolate", shared, "H4 parent alpha 0.1 all blocks", rule_mask=mask, alpha=0.1,
                     per_tensor={n: {"other": DT[n]} for n in shared})
        if k4:
            h = s.hypothesis("H4b: a second alpha 0.1 step toward the parent on the same tensors",
                             "CER -0.001 (-0.003 .. +0.002)", "interpolation (donor: parent)")
            attempt(h, "interpolate", shared, "H4b parent alpha 0.1 again", rule_mask=mask, alpha=0.1,
                    per_tensor={n: {"other": DT[n]} for n in shared})
        else:
            conv = [n for n in shared if n.startswith("C_")]
            h = s.hypothesis(f"H4b: conv stack only ({len(conv)} tensors, the block that moved most from the parent, "
                             "C_3 bias cosine 0.68) toward the parent at alpha 0.2",
                             "CER 0.000 (-0.003 .. +0.003)", "interpolation (donor: parent)")
            attempt(h, "interpolate", conv, "H4b parent alpha 0.2 conv only", rule_mask=mask, alpha=0.2,
                    per_tensor={n: {"other": DT[n]} for n in conv})

# ---- H5: r bias +0.25 ----------------------------------------------------------------------------------
R = cm.get("r")
if R is not None:
    h = s.hypothesis("H5: r bias +0.25; r deleted 23x, read as n 23x and e 19x, second-weakest letter row",
                     "CER -0.001 (-0.003 .. +0.001)", "output bias shift")
    attempt(h, "add", [BIAS], "H5 r bias +0.25", delta=bias_delta(R, 0.25))
else:
    s.note("H5 skipped: no single-label r row")

# ---- H6 (last pass): blank refinement in the kept direction, else whole-output-weight sharpening -------
if blank_steps_kept:
    if len(blank_steps_kept) >= 2:
        step = -0.5
    elif blank_steps_kept[0] == -0.5:
        step = -0.25
    else:
        step = -0.125
    h = s.hypothesis(f"H6: blank bias {step:+.3f} (refinement after kept steps {blank_steps_kept}, total so far {blank_total:+.2f})",
                     "CER -0.001 (-0.003 .. +0.001)", "output bias shift")
    attempt(h, "add", [BIAS], f"H6 blank bias {step:+.3f}", delta=bias_delta(BLANK, step))
else:
    h = s.hypothesis("H6: scale the whole O_18.lin.weight by 1.05 (evidence against the biases; no blank step was kept)",
                     "CER -0.001 (-0.003 .. +0.002)", "output weight scaling")
    attempt(h, "scale", [WEIGHT], "H6 O_18 weight x1.05", factor=1.05)

summary = s.freeze()
print("frozen:", summary["used"], "passes,", summary["edits"], "edits,", summary["kept"], "kept;",
      "last CER", (summary["last_pass"] or {}).get("cer"))
