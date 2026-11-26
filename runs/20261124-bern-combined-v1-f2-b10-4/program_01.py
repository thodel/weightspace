"""program_01 — passes 1-4: C0, H1 (toward the parent), H2 (blank bias), H3 (toward the sibling).

Hypotheses and predictions: preregistration.md section 5.
"""
import sys

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

M = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-"
W, B = "O_18.lin.weight", "O_18.lin.bias"

s = open_session(".")
print("budget left:", s.budget_left)
T = s.subject.tensors()
cm = s.subject.codec_map()
hertwig = np.array(["Hertwig" in it["key"] for it in s.practice])
print("lines:", len(s.practice), "Hertwig:", int(hertwig.sum()))

BODY = [n for n in T if not n.startswith("O_18") and "bias_ih" not in n]


def hand_cer(lines):
    a = np.array(lines, dtype=float)
    return (a[hertwig, 0].sum() / a[hertwig, 1].sum(), a[~hertwig, 0].sum() / a[~hertwig, 1].sum())


def report(label, m):
    e = s._log[-1]
    h, sr = hand_cer(s._last_lines)
    vk = e.get("vs_kept") or {}
    print(f"[{label}] CER {m.cer:.4f}  Hertwig {h:.4f}  Schlussred. {sr:.4f}  collapsed {m.collapsed}  "
          f"length ratio {m.length_ratio:.3f}  edits {m.char_edits}")
    print(f"[{label}] vs kept: delta {vk.get('delta_cer'):+.4f}  ci95 {vk.get('ci95')}")
    print(f"[{label}] confusions: {m.confusions}")
    b = m.blocks or {}
    print(f"[{label}] blocks: {b}")
    return h, sr


def donor_targets(donor):
    """Per-tensor 'other' arrays in the subject's layout. Output rows are paired by character
    (and the blank, row 0, with row 0); rows the donor lacks keep the subject's current values."""
    D = donor.tensors()
    dm = donor.codec_map()
    cur = s.subject.tensors()
    out = {}
    for n in BODY:
        assert D[n].shape == cur[n].shape, n
        out[n] = {"other": D[n].copy()}
    w, b = cur[W].copy(), cur[B].copy()
    w[0], b[0] = D[W][0], D[B][0]
    paired = 0
    for ch, i in cm.items():
        j = dm.get(ch)
        if j is not None:
            w[i], b[i] = D[W][j], D[B][j]
            paired += 1
    print("rows paired by character:", paired, "of", len(cm), "(+ blank)")
    out[W] = {"other": w}
    out[B] = {"other": b}
    return out


def decide(label, m, kept_collapsed):
    """Revert if the collapsed count rose; otherwise the harness's rule decides."""
    if m.collapsed > kept_collapsed:
        s.revert(f"{label}: collapsed rose {kept_collapsed} -> {m.collapsed}")
        print(f"[{label}] REVERTED (collapsed rose)")
        return False, kept_collapsed
    kept = s.keep(f"{label}: paired interval below 0, collapsed {m.collapsed} <= {kept_collapsed}")
    print(f"[{label}] kept = {kept}")
    return kept, (m.collapsed if kept else kept_collapsed)


# ---- pass 1: C0 -------------------------------------------------------------------------
m0 = s.measure("C0")
report("C0", m0)
if abs(m0.cer - 0.1456) > 0.0002:
    s.note(f"C0 does not reproduce c0_c1.md: {m0.cer:.4f} against 0.1456; stopping before any edit")
    print("STOP: C0 does not reproduce 0.1456")
    sys.exit(0)
kc = m0.collapsed

# ---- pass 2: H1, toward the parent, alpha 0.25 -------------------------------------------
parent = KrakenSubject(M + "hertwig-v2.mlmodel")
h = s.hypothesis(
    "H1: the Hertwig loss came with the step parent -> subject (mismatched Hertwig pairs). A quarter of the way "
    "back to hertwig-v2 in all 28 trained tensors (output rows by character, the rows of 1/2, f-hook and ring "
    "untouched) regains Hertwig faster than it loses Schlussredaktion. The parent was trained on the 103 Hertwig "
    "practice lines, so part of a practice gain is its memory of them.",
    "CER -0.006 (range -0.015 to +0.003); Hertwig 0.205 -> 0.165-0.185, Schlussredaktion 0.118 -> 0.120-0.130; "
    "collapsed 29 +/- 2",
    "interpolate toward donor (parent), all tensors")
tg = donor_targets(parent)
s.edit("interpolate", BODY + [W, B], hypothesis=h, per_tensor=tg, alpha=0.25)
m1 = s.measure("H1 parent alpha=0.25 all tensors")
report("H1", m1)
k1, kc = decide("H1", m1, kc)

# ---- pass 3: H2, blank bias -0.25 ---------------------------------------------------------
h = s.hypothesis(
    "H2: deletions dominate and the subject's blank bias (+0.998) is the highest of the family (parent +0.879). "
    "Lowering it by 0.25 turns marginal blank frames into characters.",
    "CER -0.002 (range -0.006 to +0.003); fewer deletions, more insertions, length ratio up; "
    "the interval contains 0 more likely than not",
    "bias shift, blank row")
d = np.zeros_like(s.subject.tensors()[B])
d[0] = -0.25
s.edit("add", [B], hypothesis=h, delta=d)
m2 = s.measure("H2 blank bias -0.25")
report("H2", m2)
k2, kc = decide("H2", m2, kc)

# ---- pass 4: H3, toward the sibling, alpha 0.30 ---------------------------------------------
sibling = KrakenSubject(M + "schlussredaktion-v1.mlmodel")
h = s.hypothesis(
    "H3: the sibling is the clean fine-tune of the same parent on Schlussredaktion (69 % of the practice "
    "characters; it was trained on the 197 Schlussredaktion practice lines, the subject too). Averaging 0.7/0.3 "
    "with it in all 28 trained tensors (rows by character, the f-hook row untouched) lowers the Schlussredaktion "
    f"CER and costs little on Hertwig. State before: H1 kept={k1}, H2 kept={k2}.",
    "CER -0.002 (range -0.006 to +0.004); Schlussredaktion -0.003 to -0.008, Hertwig +0.000 to +0.010",
    "interpolate toward donor (sibling), all tensors")
tg = donor_targets(sibling)
s.edit("interpolate", BODY + [W, B], hypothesis=h, per_tensor=tg, alpha=0.30)
m3 = s.measure("H3 sibling alpha=0.30 all tensors")
report("H3", m3)
k3, kc = decide("H3", m3, kc)

print(f"SUMMARY program_01: H1 kept={k1}  H2 kept={k2}  H3 kept={k3}  kept collapsed={kc}  budget left={s.budget_left}")
