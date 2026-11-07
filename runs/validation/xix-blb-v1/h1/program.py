"""Programmer's script, kraken-german-xix-blb-v1 (session h1). Written before any pass.

Greedy and sequential: each edit is measured once and kept or reverted by the harness's rule.
Uses only the wsp API and numpy; the donor (parent) is read-only.
"""
import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows

PARENT_ID = "kraken-german-xix-v1"
PARENT_PATH = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"
BLANK = 0
LETTERS = "eintrs"

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
SPACE = cm.get(" ")
LETTER_ROWS = [cm[c] for c in LETTERS if c in cm]

base = s.measure("C0 baseline")
print(f"C0 CER {base.cer:.4f} collapsed {base.collapsed}")
state = {"cer": base.cer}


def try_edit(label, hyp_text, prediction, edit_class, apply, mask=None):
    """One hypothesis, one edit, one measured pass, then the keep rule. Returns whether it was kept."""
    h = s.hypothesis(hyp_text, prediction, edit_class)
    apply(h)
    m = s.measure(label, rule_mask=mask)
    kept = s.keep(label)
    print(f"{label}: CER {m.cer:.4f} (vs last kept {m.cer - state['cer']:+.4f}) kept={kept} collapsed={m.collapsed}")
    if kept:
        state["cer"] = m.cer
    return kept


def blank_step(step):
    def apply(h):
        d = np.zeros_like(T[B])
        d[BLANK] = step
        s.edit("add", [B], hypothesis=h, delta=d)
    return apply


def space_step(step):
    def apply(h):
        d = np.zeros_like(T[B])
        d[SPACE] = step
        s.edit("add", [B], hypothesis=h, delta=d)
    return apply


def letter_step(frac):
    def apply(h):
        d = np.zeros_like(T[W])
        d[LETTER_ROWS] = frac * T[W][LETTER_ROWS]
        s.edit("add", [W], hypothesis=h, delta=d)
    return apply


# H1: lower the blank bias.
if try_edit("H1 blank bias -0.3",
            "lower the CTC blank bias so fewer frames are blank; deletions dominate the C0 errors",
            "CER down about 0.010; length ratio up toward 1.0",
            "scalar output bias (add on row 0)", blank_step(-0.3)):
    # H1b: only if H1 was kept, one more step.
    try_edit("H1b blank bias -0.3 more",
             "a second blank step still lowers CER, with diminishing returns",
             "CER down about 0.005; insertions start to rise",
             "scalar output bias (add on row 0)", blank_step(-0.3))
else:
    s.note("H1 refused: no second blank step")

# H2: raise the space row bias (space deletions dominate the space errors).
if SPACE is not None:
    try_edit("H2 space bias +1.5",
             "the space row at -7.85 is too deep; a bias step recovers missed spaces",
             "CER down about 0.004; space deletions fall, space insertions rise slightly",
             "scalar output bias (add on space row)", space_step(1.5))
else:
    s.note("H2 skipped: no space in the codec")

# H3: enlarge the weakest letter rows (about x1.1).
if LETTER_ROWS:
    try_edit("H3 letter rows x1.1",
             "the weak letter rows e,i,n,r,t,s lose outputs; a larger row norm recovers them",
             "CER down about 0.003 (sign uncertain); deletions of these letters fall",
             "row-wise weight edit (add 0.1 times the row)", letter_step(0.1))

# H4: pull the output layer toward the parent (negative control; donor rule applies).
donor = KrakenSubject(PARENT_PATH)
dcm = donor.codec_map()
rmap = codec_row_map(cm, dcm)
rmap[BLANK] = BLANK
DT = donor.tensors()


def donor_step(alpha):
    def apply(h):
        Wd = gather_rows(DT[W], rmap, T[W].shape[0], fill=T[W].copy())
        Bd = gather_rows(DT[B], rmap, T[B].shape[0], fill=T[B].copy())
        s.edit("interpolate", [W, B], hypothesis=h,
               per_tensor={W: {"other": Wd}, B: {"other": Bd}}, alpha=alpha)
    return apply


mask = s.unseen_by(PARENT_ID)
if sum(mask) == 0:
    s.note(f"H4 skipped: donor rule mask empty for {PARENT_ID}")
else:
    try_edit("H4 output layer toward parent alpha 0.05",
             "the parent's output rows are worse on these lines; a small pull should not help",
             "CER rises 0.000 to +0.01; keep refused",
             "interpolation with donor (alpha 0.05), donor rule mask",
             donor_step(0.05), mask=mask)

summary = s.freeze()
print("frozen: used", summary["used"], "edits", summary["edits"], "kept", summary["kept"],
      "final CER on practice", summary["last_pass"]["cer"] if summary.get("last_pass") else None)
