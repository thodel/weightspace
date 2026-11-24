"""program_01 — round 1 of session 20261124-bern-combined-v1-f2-b10-3.

Pass 1: C0 (baseline, must give CER 0.1456).
Pass 2: H1 — all trained tensors 30 % of the way toward the parent hertwig-v2.
Pass 3: H2 — a branch on the H1 decision (see preregistration.md section 5).
No freeze here.
"""
import sys

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"

s = open_session(".")
print("budget left at start:", s.budget_left)
T = s.subject.tensors()
cm = s.subject.codec_map()
items = list(s.practice)
hand = ["H" if "Hertwig" in it["key"] else "S" for it in items]


def per_hand(lines):
    out = {}
    for h in "HS":
        e = sum(l[0] for l, hh in zip(lines, hand) if hh == h)
        c = sum(l[1] for l, hh in zip(lines, hand) if hh == h)
        out[h] = e / c
    return out


def report(label, m):
    ph = per_hand(s._last_lines)
    e = s._log[-1]
    print(f"[{label}] CER {m.cer:.4f} (Hertwig {ph['H']:.4f}, Schlussred. {ph['S']:.4f}) "
          f"edits {m.char_edits} collapsed {m.collapsed} length_ratio {m.length_ratio:.3f}")
    print(f"    vs_kept delta {e['vs_kept']['delta_cer']:+.4f} ci95 "
          f"[{e['vs_kept']['ci95'][0]:+.4f}, {e['vs_kept']['ci95'][1]:+.4f}]; "
          f"vs_baseline delta {e['vs_baseline']['delta_cer']:+.4f}")
    print("    blocks:", {k: v for k, v in m.blocks.items()} if isinstance(m.blocks, dict) else m.blocks)
    print("    confusions:", m.confusions)
    return ph


# ---------------------------------------------------------------- pass 1: C0
m0 = s.measure("C0")
report("C0", m0)
if abs(m0.cer - 0.1456) > 0.0005 or m0.collapsed != 29:
    print("STOP: C0 does not reproduce c0_c1.md (CER 0.1456, 29 collapsed). No edit made.")
    sys.exit(0)
kept_collapsed = m0.collapsed

# ---------------------------------------------------------------- donor, aligned by character
donor = KrakenSubject(PARENT)
D = donor.tensors()
dcm = donor.codec_map()
rowmap = {cm[ch]: dcm[ch] for ch in cm if ch in dcm}     # subject row -> parent row, by character
rowmap[0] = 0                                            # CTC blank
only_subject = [ch for ch in cm if ch not in dcm]
print("rows aligned by character:", len(rowmap), "; subject-only characters (left untouched):", only_subject)

NAMES = [n for n in T if "bias_ih" not in n]             # bias_ih is all zero in every model


def parent_like(name):
    """The parent's tensor in the subject's shape. Output rows are placed by character;
    rows the parent lacks take the subject's CURRENT values, so interpolation leaves them as they are."""
    cur = s.subject.tensors()[name]
    if not name.startswith("O_18"):
        assert D[name].shape == cur.shape, name
        return D[name]
    out = cur.copy()
    for ia, ib in rowmap.items():
        out[ia] = D[name][ib]
    return out


def toward_parent(alpha, hyp):
    s.edit("interpolate", NAMES, hypothesis=hyp, alpha=float(alpha),
           per_tensor={n: {"other": parent_like(n)} for n in NAMES})


def decide(label, m):
    """Keep rule of the harness, plus: revert if the collapsed count rose against the last kept state."""
    global kept_collapsed
    if m.collapsed > kept_collapsed:
        s.revert(f"{label}: collapsed rose {kept_collapsed} -> {m.collapsed}")
        print(f"    {label}: REVERTED (collapsed rose {kept_collapsed} -> {m.collapsed})")
        return False
    kept = s.keep(f"{label}: paired 95 % interval below 0, collapsed not up")
    print(f"    {label}: {'KEPT' if kept else 'REFUSED by the keep rule'}")
    if kept:
        kept_collapsed = m.collapsed
    return kept


# ---------------------------------------------------------------- pass 2: H1
h1 = s.hypothesis(
    "H1: the subject forgot part of the parent's Hertwig reading (0.205 against the parent's 0.114 on the 103 "
    "Hertwig lines, which the parent was trained on with correct images). All three models lie in one basin "
    "(cosine to parent >= 0.96 per tensor), so moving every trained tensor 30 % of the way back to hertwig-v2 "
    "(output rows by character, the three subject-only rows untouched) recovers part of it at a small cost on "
    "Schlussredaktion.",
    "practice CER -0.008 (range -0.003 to -0.015): Hertwig -0.030, Schlussredaktion +0.003; collapsed 29 +- 2",
    "interpolate toward parent, all tensors")
toward_parent(0.30, h1)
m1 = s.measure("H1 parent a=0.30 all tensors")
ph1 = report("H1", m1)
d1 = s._log[-1]["vs_kept"]["delta_cer"]
k1 = decide("H1", m1)

# ---------------------------------------------------------------- pass 3: H2 (branch)
if k1:
    h2 = s.hypothesis(
        "H2a (H1 kept): the optimum along subject->parent lies beyond 0.30. A further step of 2/7 from the kept "
        "state puts every tensor at 50 % subject / 50 % parent.",
        "against the kept H1 state: CER -0.002 (range -0.008 to +0.006): Hertwig -0.020, Schlussredaktion "
        "+0.007; more likely refused than kept",
        "interpolate toward parent, all tensors")
    toward_parent(2.0 / 7.0, h2)
    lab = "H2a parent total a=0.50"
elif d1 < 0:
    h2 = s.hypothesis(
        "H2b (H1 lowered CER but its interval reached 0): the direction is right and the step too small for the "
        "rule; 50 % of the way to the parent from the unedited subject.",
        "against C0: CER -0.010 (range -0.020 to +0.005): Hertwig -0.050, Schlussredaktion +0.008",
        "interpolate toward parent, all tensors")
    toward_parent(0.50, h2)
    lab = "H2b parent a=0.50"
else:
    h2 = s.hypothesis(
        "H2c (H1 raised CER or the collapsed count): 30 % was too far or the path has a barrier; 12 % of the "
        "way to the parent from the unedited subject.",
        "against C0: CER -0.002 (range -0.006 to +0.002); expected refused",
        "interpolate toward parent, all tensors")
    toward_parent(0.12, h2)
    lab = "H2c parent a=0.12"
m2 = s.measure(lab)
report(lab, m2)
decide(lab.split()[0], m2)

print("budget left:", s.budget_left, "; kept collapsed:", kept_collapsed)
print("digest:", s.subject.digest())
