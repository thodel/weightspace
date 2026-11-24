"""program_02 — passes 5-8: F1 (parent, all tensors), F2a (lower blocks), F2b (upper blocks), F4 (blank).

Hypotheses and predictions: preregistration.md section 5, round 1. Starts from the kept state
of program_01 (H1 + H2: CER 0.1364, collapsed 21). No freeze here.
"""
import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

M = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-"
W, B = "O_18.lin.weight", "O_18.lin.bias"
KEPT_COLLAPSED = 21          # pass 3 (H2), the last kept state

s = open_session(".")
print("budget left:", s.budget_left)
T = s.subject.tensors()
cm = s.subject.codec_map()
hertwig = np.array(["Hertwig" in it["key"] for it in s.practice])
print("blank bias of the restored state:", float(T[B][0]), "(expected about 0.718)")

BODY = [n for n in T if not n.startswith("O_18") and "bias_ih" not in n]
LOWER = [n for n in BODY if n.split(".")[0] in ("C_0", "C_3", "C_6", "C_9", "L_12")]
UPPER = [n for n in BODY if n.split(".")[0] in ("L_14", "L_16")]
assert len(BODY) == 26 and len(LOWER) == 14 and len(UPPER) == 12, (len(BODY), len(LOWER), len(UPPER))


def hand_cer(lines):
    a = np.array(lines, dtype=float)
    return (a[hertwig, 0].sum() / a[hertwig, 1].sum(), a[~hertwig, 0].sum() / a[~hertwig, 1].sum())


def report(label, m):
    vk = s._log[-1].get("vs_kept") or {}
    h, sr = hand_cer(s._last_lines)
    print(f"[{label}] CER {m.cer:.4f}  Hertwig {h:.4f}  Schlussred. {sr:.4f}  collapsed {m.collapsed}  "
          f"length ratio {m.length_ratio:.3f}  edits {m.char_edits}")
    print(f"[{label}] vs kept: delta {vk.get('delta_cer'):+.4f}  ci95 {vk.get('ci95')}")
    print(f"[{label}] confusions: {m.confusions}")
    for name, b in (m.blocks or {}).get("blocks", {}).items():
        print(f"[{label}]   {name}: sub {b['sub']} del {b['del']} ins {b['ins']}")


def parent_targets(donor, names, output):
    """'other' arrays in the subject's layout. Output rows paired by character; rows the parent
    lacks and the blank BIAS keep the current values (the blank weight row does move)."""
    D = donor.tensors()
    dm = donor.codec_map()
    cur = s.subject.tensors()
    out = {n: {"other": D[n].copy()} for n in names}
    if output:
        w, b = cur[W].copy(), cur[B].copy()
        w[0] = D[W][0]
        for ch, i in cm.items():
            j = dm.get(ch)
            if j is not None:
                w[i], b[i] = D[W][j], D[B][j]
        assert b[0] == cur[B][0]
        out[W] = {"other": w}
        out[B] = {"other": b}
    return out


def decide(label, m, kept_collapsed):
    if m.collapsed > kept_collapsed:
        s.revert(f"{label}: collapsed rose {kept_collapsed} -> {m.collapsed}")
        print(f"[{label}] REVERTED (collapsed rose)")
        return False, kept_collapsed
    kept = s.keep(f"{label}: paired interval below 0, collapsed {m.collapsed} <= {kept_collapsed}")
    print(f"[{label}] kept = {kept}")
    return kept, (m.collapsed if kept else kept_collapsed)


kc = KEPT_COLLAPSED
parent = KrakenSubject(M + "hertwig-v2.mlmodel")
NOTE = ("The parent was trained on the 103 Hertwig practice lines with correct images, so part of a practice "
        "gain is its memory of them. The blank bias stays at its kept value.")

# ---- pass 5: F1, all tensors, a further 0.20 toward the parent ----------------------------
h = s.hypothesis(
    "F1: line search. H1 (0.25 toward the parent) was kept; a further 0.20 of the remaining way in all 28 "
    "trained tensors (about 0.40 in total) still gains more on Hertwig than it costs on Schlussredaktion. " + NOTE,
    "CER -0.001 (range -0.007 to +0.005); Hertwig -0.010 to -0.020, Schlussredaktion +0.006 to +0.012; "
    "interval expected to contain 0 (kept with probability about 0.3)",
    "interpolate toward donor (parent), all tensors")
names = BODY + [W, B]
s.edit("interpolate", names, hypothesis=h, per_tensor=parent_targets(parent, BODY, True), alpha=0.20)
m = s.measure("F1 parent alpha=0.20 all tensors, blank bias fixed")
report("F1", m)
k1, kc = decide("F1", m, kc)

# ---- pass 6: F2a, lower blocks, a further 0.25 --------------------------------------------
h = s.hypothesis(
    "F2a: the Hertwig damage sits more in the lower blocks (C_0 C_3 C_6 C_9 L_12), where the subject's and the "
    "sibling's deltas from the parent differ most (cosine 0.24-0.48). A further 0.25 toward the parent there "
    f"only. State before: F1 kept={k1}. " + NOTE,
    "CER -0.001 (range -0.006 to +0.005); Hertwig -0.005 to -0.015; a smaller Schlussredaktion cost than a "
    "global step of the same size",
    "interpolate toward donor (parent), lower blocks")
s.edit("interpolate", LOWER, hypothesis=h, per_tensor=parent_targets(parent, LOWER, False), alpha=0.25)
m = s.measure("F2a parent alpha=0.25 C_* and L_12")
report("F2a", m)
k2a, kc = decide("F2a", m, kc)

# ---- pass 7: F2b, upper blocks, a further 0.25 --------------------------------------------
h = s.hypothesis(
    "F2b: the counterpart of F2a. A further 0.25 toward the parent in the upper blocks only (L_14 L_16 O_18, "
    f"rows by character, blank bias untouched). State before: F1 kept={k1}, F2a kept={k2a}. " + NOTE,
    "CER -0.001 (range -0.006 to +0.005); if the hands separate by depth, exactly one of F2a and F2b is kept",
    "interpolate toward donor (parent), upper blocks")
names = UPPER + [W, B]
s.edit("interpolate", names, hypothesis=h, per_tensor=parent_targets(parent, UPPER, True), alpha=0.25)
m = s.measure("F2b parent alpha=0.25 L_14 L_16 O_18, blank bias fixed")
report("F2b", m)
k2b, kc = decide("F2b", m, kc)

# ---- pass 8: F4, blank bias a further -0.25 ------------------------------------------------
h = s.hypothesis(
    "F4: H2 (blank bias -0.25) was kept with a small gain. A further -0.25 turns more marginal blank frames "
    f"into characters. State before: F1 kept={k1}, F2a kept={k2a}, F2b kept={k2b}.",
    "CER -0.001 (range -0.003 to +0.002); deletions down, insertions up; kept with probability about 0.35",
    "bias shift, blank row")
d = np.zeros_like(s.subject.tensors()[B])
d[0] = -0.25
s.edit("add", [B], hypothesis=h, delta=d)
m = s.measure("F4 blank bias a further -0.25")
report("F4", m)
k4, kc = decide("F4", m, kc)

print(f"SUMMARY program_02: F1 kept={k1}  F2a kept={k2a}  F2b kept={k2b}  F4 kept={k4}  "
      f"kept collapsed={kc}  blank bias={float(s.subject.tensors()[B][0]):.3f}  budget left={s.budget_left}")
