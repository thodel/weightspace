"""Attempt o1 on kraken-german-xix-blb-v1 (programmer: claude-opus-5-5). Written in advance; wsp + numpy only.

Plan: blank bias down (H1), space bias up (H2), task-vector extrapolation away from the parent (H3, donor rule),
bias up on the most-deleted letters (H4). At most 9 passes including the baseline; every edit decided by s.keep().
"""
import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT_ID = "kraken-german-xix-v1"
PARENT_PATH = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
B = "O_18.lin.bias"
W = "O_18.lin.weight"

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
n_rows = T[B].shape[0]


def delta_vs_kept():
    v = s._vs_kept or {}
    return float(v.get("delta_cer", 0.0))


def bias_delta(rows_vals):
    d = np.zeros(n_rows, dtype=np.float32)
    for r, v in rows_vals:
        d[r] += v
    return d


def try_edit(text, prediction, edit_class, op, names, label, reason, rule_mask=None, per_tensor=None, **kw):
    """One edit, one pass, one keep decision. Returns (kept, delta_cer vs last kept) or (None, None) without budget."""
    if s.budget_left < 1:
        s.note(f"skipped '{label}': no passes left")
        return None, None
    h = s.hypothesis(text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kw)
    m = s.measure(label, rule_mask=rule_mask)
    d = delta_vs_kept()
    kept = s.keep(reason)
    s.note(f"{label}: CER {m.cer:.4f}, delta vs kept {d:+.4f}, kept={kept}")
    return kept, d


# ---- pass 1: C0 baseline -------------------------------------------------
m0 = s.measure("C0 baseline")
s.note(f"C0 CER {m0.cer:.4f}")

# ---- H1: blank bias down -------------------------------------------------
kept, d = try_edit("Deletions dominate (length ratio 0.97); blank bias +1.23 wins frames of weak letters. Lower it by 0.5.",
                   "CER down 0.3-1.0 points", "bias (blank)", "add", [B], "H1 blank bias -0.5",
                   "blank -0.5 reduces deletions", delta=bias_delta([(0, -0.5)]))
if kept:
    kept2, _ = try_edit("Further blank bias -0.5 (cumulative -1.0).", "CER down 0.2-0.5 points more",
                        "bias (blank)", "add", [B], "H1 blank bias -1.0 total", "blank -1.0 total",
                        delta=bias_delta([(0, -0.5)]))
    if kept2:
        try_edit("Further blank bias -0.5 (cumulative -1.5).", "CER down 0-0.3 points more; may turn into insertions",
                 "bias (blank)", "add", [B], "H1 blank bias -1.5 total", "blank -1.5 total",
                 delta=bias_delta([(0, -0.5)]))
elif d is not None and d < 0:
    try_edit("Blank -0.5 moved CER down but not significantly; try -1.0 directly.", "CER down 0.3-1.0 points",
             "bias (blank)", "add", [B], "H1b blank bias -1.0", "blank -1.0", delta=bias_delta([(0, -1.0)]))
else:
    s.note("H1 failed with CER up: deletions are not a blank-margin problem; blank bias left alone.")

# ---- H2: space bias up ---------------------------------------------------
if " " in cm:
    sp = cm[" "]
    kept, _ = try_edit("78 space deletions vs 30 spurious spaces; space bias -7.85 is the lowest. Raise by 0.5.",
                       "CER down 0.1-0.4 points", "bias (space)", "add", [B], "H2 space bias +0.5",
                       "space +0.5", delta=bias_delta([(sp, 0.5)]))
    if kept:
        try_edit("Further space bias +0.5 (cumulative +1.0).", "CER down 0-0.3 points more",
                 "bias (space)", "add", [B], "H2 space bias +1.0 total", "space +1.0 total",
                 delta=bias_delta([(sp, 0.5)]))
else:
    s.note("no single-label space in codec; H2 skipped")

# ---- H3: task-vector extrapolation away from the parent (donor rule) ------
donor = KrakenSubject(PARENT_PATH)
DT = donor.tensors()
dcm = donor.codec_map()
mask = s.unseen_by(PARENT_ID)


def parent_targets():
    """Parent arrays in the subject's current shapes; output rows aligned by character, child-only rows = current values."""
    cur = s.subject.tensors()
    out = {}
    for n, v in cur.items():
        if n in (W, B):
            arr = v.copy()
            pairs = [(0, 0)] + [(r, dcm[ch]) for ch, r in cm.items() if ch in dcm and r != 0]
            for r, rd in pairs:
                arr[r] = DT[n][rd]
            out[n] = {"other": arr.astype(np.float32)}
        elif n in DT and DT[n].shape == v.shape:
            out[n] = {"other": DT[n].astype(np.float32)}
    return out


names = list(parent_targets().keys())
kept, d = try_edit("Parent CER 0.327 vs subject 0.138: the fine-tune direction helps on BLB hands; extrapolate it by 10 % "
                   "(interpolate alpha -0.1, all tensors, output rows aligned by character).",
                   "CER down 0.1-0.5 points", "donor (task-vector extrapolation)", "interpolate", names,
                   "H3 extrapolate alpha -0.1", "extrapolation along fine-tune direction", rule_mask=mask,
                   per_tensor=parent_targets(), alpha=-0.1)
if kept:
    try_edit("Extrapolate a further 10 % along the fine-tune direction.", "CER down 0-0.3 points more",
             "donor (task-vector extrapolation)", "interpolate", names, "H3 extrapolate further alpha -0.1",
             "further extrapolation", rule_mask=mask, per_tensor=parent_targets(), alpha=-0.1)
elif d is not None and d > 0:
    try_edit("Extrapolation hurt; the opposite small step toward the parent (WiSE-FT style) may regularise for unseen hands.",
             "CER down 0-0.3 points (or up if the subject is at its optimum)", "donor (interpolation toward parent)",
             "interpolate", names, "H3b interpolate alpha +0.05", "regularisation toward parent",
             rule_mask=mask, per_tensor=parent_targets(), alpha=0.05)
else:
    s.note("H3 extrapolation flat; not pursued further.")

# ---- H4: bias up on the most-deleted letters -----------------------------
letters = [c for c in "einrst" if c in cm]
try_edit("e,i,n,r,s,t are the most-deleted letters and have the weakest output rows; +0.3 bias on each.",
         "CER down 0.1-0.4 points", "bias (letters)", "add", [B], "H4 letter bias +0.3",
         "letters +0.3", delta=bias_delta([(cm[c], 0.3) for c in letters]))

summary = s.freeze()
print({k: summary[k] for k in ("used", "edits", "kept", "baseline", "last_pass")})
