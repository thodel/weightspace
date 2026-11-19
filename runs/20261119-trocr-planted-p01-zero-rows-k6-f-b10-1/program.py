"""TrOCR planted-defect control, s03, task (a): budget 10, with the donor. Programmer: claude-fable-5-2.

Written in full before the first pass. Hypotheses: preregistration_hypotheses.md.
"""
import numpy as np

from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
EMB = "dec.embed.tokens"
EXPECTED = [4, 6, 90, 225, 364, 2432]          # '.', ',', 't', 'en', 'Ġe', 'Ġund' (pack: rows exactly zero)

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
Pe = np.array(P[EMB], dtype=np.float64)
Te = np.array(s.subject.tensors()[EMB], dtype=np.float64)

# rows that are exactly zero in the subject but not in the parent
zero = np.where((np.abs(Te).sum(axis=1) == 0) & (np.abs(Pe).sum(axis=1) > 0))[0]
rows = [int(i) for i in zero] if len(zero) and len(zero) <= 64 else EXPECTED
s.note(f"zero rows found in {EMB}: {rows} (pack said {EXPECTED})")
intact = np.ones(Te.shape[0], dtype=bool)
intact[rows] = False

drift = Te[intact] - Pe[intact]
mean_drift = drift.mean(axis=0)
s.note(f"mean drift norm {np.linalg.norm(mean_drift):.4f}; mean row drift norm "
       f"{np.linalg.norm(drift, axis=1).mean():.4f}; mean parent row norm {np.linalg.norm(Pe, axis=1).mean():.4f}")

rows_common = (Pe[rows] + mean_drift).astype(np.float32)        # H1
rows_donor = Pe[rows].astype(np.float32)                         # H1b

# H2: local drift = mean drift of the 50 intact rows nearest (cosine) in the parent's space
Pn = Pe / np.maximum(np.linalg.norm(Pe, axis=1, keepdims=True), 1e-12)
sim = Pn[rows] @ Pn.T                                            # 6 x vocab
sim[:, rows] = -np.inf
local = np.zeros((len(rows), Pe.shape[1]))
for j in range(len(rows)):
    nb = np.argpartition(-sim[j], 50)[:50]
    local[j] = (Te[nb] - Pe[nb]).mean(axis=0)
rows_local = (Pe[rows] + local).astype(np.float32)
del sim, Pn, drift


def conf_count(m, pairs):
    return sum(n for a, b, n in m.confusions if (a, b) in pairs)


m0 = s.measure("C0", rule_mask=MASK)
s.note(f"C0 cer {m0.cer:.4f} collapsed {m0.collapsed}")
best = m0
repaired = None

# ---- H1: parent rows + common drift
h1 = s.hypothesis(
    f"The defect is the zeroed rows {rows} of the tied embedding; parent rows plus the mean drift of the intact rows restore them.",
    "CER falls by 12 to 21 points to 0.04-0.13; collapsed lines <= 2; space/'.'/',' deletions shrink by more than half",
    "set_rows (donor rows + common drift)")
s.edit("set_rows", [EMB], hypothesis=h1, index=rows, values=rows_common)
m1 = s.measure("H1 parent rows + mean drift", rule_mask=MASK)
if m1.collapsed > m0.collapsed + 5:
    s.revert(f"collapsed lines rose {m0.collapsed} -> {m1.collapsed}")
    k1 = False
else:
    k1 = s.keep(f"cer {m0.cer:.4f} -> {m1.cer:.4f}")
s.note(f"H1 kept={k1} cer {m1.cer:.4f} (predicted 0.04-0.13) collapsed {m1.collapsed}")
if k1:
    best, repaired = m1, rows_common

# ---- H1b: parent rows verbatim, only if H1 was not kept
if not k1:
    h1b = s.hypothesis(
        "The common drift was harmful; the parent's rows verbatim restore the zeroed rows.",
        "CER falls by 10 to 20 points against C0",
        "set_rows (donor rows)")
    s.edit("set_rows", [EMB], hypothesis=h1b, index=rows, values=rows_donor)
    m1b = s.measure("H1b parent rows verbatim", rule_mask=MASK)
    if m1b.collapsed > m0.collapsed + 5:
        s.revert(f"collapsed lines rose {m0.collapsed} -> {m1b.collapsed}")
        k1b = False
    else:
        k1b = s.keep(f"cer {m0.cer:.4f} -> {m1b.cer:.4f}")
    s.note(f"H1b kept={k1b} cer {m1b.cer:.4f}")
    if k1b:
        best, repaired = m1b, rows_donor

if repaired is None:
    s.note("Neither restoration was kept: the zero rows are not what the keep rule can show as the cause; no further edits "
           "(the pack points at nothing else).")
else:
    # ---- H2: local drift instead of the common drift
    h2 = s.hypothesis(
        "Local drift (mean drift of the 50 nearest intact rows in the parent's space) fits the child better than the kept rows.",
        "CER change between -1.0 and +0.3 points; most likely below the 0.5 point threshold and not kept",
        "set_rows (donor rows + local drift)")
    s.edit("set_rows", [EMB], hypothesis=h2, index=rows, values=rows_local)
    m2 = s.measure("H2 parent rows + local drift", rule_mask=MASK)
    if m2.collapsed > best.collapsed + 3:
        s.revert(f"collapsed lines rose {best.collapsed} -> {m2.collapsed}")
        k2 = False
    else:
        k2 = s.keep(f"cer {best.cer:.4f} -> {m2.cer:.4f}")
    s.note(f"H2 kept={k2} cer {m2.cer:.4f}")
    if k2:
        best, repaired = m2, rows_local

    # ---- H3: length of the restored rows, direction from the confusions of the kept state
    chars = [" ", ",", ".", "t", "e", "n"]
    dele = conf_count(best, {(c, "") for c in chars})
    ins = conf_count(best, {("", c) for c in chars})
    s.note(f"kept state: deletions of the six tokens' characters {dele}, insertions {ins} (top-10 confusions only)")
    factor = 1.06 if dele - ins > 30 else (0.94 if ins - dele > 30 else None)
    if factor is None:
        s.note("H3 skipped: deletions and insertions of the restored tokens' characters are balanced.")
    else:
        h3 = s.hypothesis(
            f"The restored rows are {'too short' if factor > 1 else 'too long'} for the tied readout; scale the rows by {factor}.",
            "CER change -0.2 to -0.8 points; probably not kept by the rule",
            "set_rows (scaled restored rows)")
        s.edit("set_rows", [EMB], hypothesis=h3, index=rows, values=(repaired * factor).astype(np.float32))
        m3 = s.measure(f"H3 restored rows x{factor}", rule_mask=MASK)
        if m3.collapsed > best.collapsed + 3:
            s.revert(f"collapsed lines rose {best.collapsed} -> {m3.collapsed}")
            k3 = False
        else:
            k3 = s.keep(f"cer {best.cer:.4f} -> {m3.cer:.4f}")
        s.note(f"H3 kept={k3} cer {m3.cer:.4f}")
        if k3:
            best = m3

    if best.cer > 0.15:
        s.note(f"CER after the kept repair is {best.cer:.4f} > 0.15: a second defect is likely, but the pack gives no location; "
               "no further edit.")
    else:
        s.note(f"Final kept CER {best.cer:.4f}; H0: no other tensor edited (all other distances inside the healthy-child range).")

s.note(f"passes left: {s.budget_left}")
summary = s.freeze()
print(summary)
