"""program_02 — round 2 of session 20261124-bern-combined-v1-f2-b10-3.

Kept state at start: H1 (all trained tensors 30 % toward hertwig-v2), CER 0.1381, 22 collapsed.
Pass 4: H3a — top (L_16 + O_18) back to the unedited subject.
Pass 5: H3b — bottom (conv stack + L_12) further toward the parent, to 55 % in total.
Pass 6: H5  — blank bias -0.40.
No freeze here.
"""
import json

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"
ORIGINAL = json.load(open("session.json"))["subject"]      # the unedited subject, read-only

s = open_session(".")
print("budget left at start:", s.budget_left, "; digest", s.subject.digest()[:12])
T = s.subject.tensors()
cm = s.subject.codec_map()
items = list(s.practice)
hand = ["H" if "Hertwig" in it["key"] else "S" for it in items]
kept_collapsed = 22                                         # H1, pass 2 (stdout of program_01)


def per_hand(lines):
    out = {}
    for h in "HS":
        e = sum(l[0] for l, hh in zip(lines, hand) if hh == h)
        c = sum(l[1] for l, hh in zip(lines, hand) if hh == h)
        out[h] = (e, e / c)
    return out


def report(label, m):
    ph = per_hand(s._last_lines)
    e = s._log[-1]
    print(f"[{label}] CER {m.cer:.4f} (Hertwig {ph['H'][1]:.4f} / {ph['H'][0]} edits, "
          f"Schlussred. {ph['S'][1]:.4f} / {ph['S'][0]} edits) "
          f"edits {m.char_edits} collapsed {m.collapsed} length_ratio {m.length_ratio:.3f}")
    print(f"    vs_kept delta {e['vs_kept']['delta_cer']:+.4f} ci95 "
          f"[{e['vs_kept']['ci95'][0]:+.4f}, {e['vs_kept']['ci95'][1]:+.4f}]; "
          f"vs_baseline delta {e['vs_baseline']['delta_cer']:+.4f}")
    b = m.blocks.get("blocks", {}) if isinstance(m.blocks, dict) else {}
    print("    blocks:", {k: (v["sub"], v["del"], v["ins"]) for k, v in b.items()}, "(sub, del, ins)")
    print("    confusions:", m.confusions)


def decide(label, m):
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


donor = KrakenSubject(PARENT)
D = donor.tensors()
dcm = donor.codec_map()
rowmap = {cm[ch]: dcm[ch] for ch in cm if ch in dcm}
rowmap[0] = 0
orig = KrakenSubject(ORIGINAL)
O = orig.tensors()
assert orig.codec_map() == cm

TRAINED = [n for n in T if "bias_ih" not in n]
TOP = [n for n in TRAINED if n.startswith(("L_16", "O_18"))]
BOTTOM = [n for n in TRAINED if n.startswith(("C_", "L_12"))]
print("TOP", len(TOP), "tensors; BOTTOM", len(BOTTOM), "tensors")

# check of the kept state: every trained tensor is 0.7 subject + 0.3 parent
n0 = "L_14.layer.weight_hh_l0"
print("kept state check (max |T - (0.7 orig + 0.3 parent)|, L_14 hh):",
      float(np.abs(T[n0] - (0.7 * O[n0] + 0.3 * D[n0])).max()))

# ---------------------------------------------------------------- pass 4: H3a
h = s.hypothesis(
    "H3a: the Schlussredaktion cost of H1 (+96 edits on its readable lines: D->d, lost and inserted spaces and "
    "slashes, dropped superscript e, ß->s) is the parent's transcription convention leaking in, and conventions "
    "sit in the top of the network. Setting L_16 and O_18 back to the unedited subject, with conv, L_12 and "
    "L_14 left at 30 % parent, returns part of that cost and keeps most of the Hertwig gain.",
    "against the kept H1 state: CER -0.002 (range -0.008 to +0.006): Schlussredaktion -0.005, Hertwig +0.005; "
    "about as likely refused as kept; collapsed 22 +- 2",
    "transplant (top block back to the original subject)")
s.edit("transplant", TOP, hypothesis=h, per_tensor={n: {"donor": O[n]} for n in TOP})
m = s.measure("H3a top L_16+O_18 back to subject")
report("H3a", m)
k3a = decide("H3a", m)

# ---------------------------------------------------------------- pass 5: H3b
h = s.hypothesis(
    "H3b: the Hertwig gain of H1 comes from the visual blocks, where subject and sibling task vectors agree "
    "least (cosine 0.24-0.48). Moving conv stack and L_12 further toward the parent, from 30 % to 55 % "
    f"(a step of 0.25/0.7 on these tensors only; H3a {'kept' if k3a else 'not kept'}), gains on Hertwig "
    "without the convention cost that the whole-model step to 50 % had (+0.025 on Schlussredaktion).",
    "against the kept state: CER -0.002 (range -0.008 to +0.008): Hertwig -0.015, Schlussredaktion +0.004; "
    "more likely refused than kept",
    "interpolate toward parent, conv + L_12")
s.edit("interpolate", BOTTOM, hypothesis=h, alpha=0.25 / 0.7,
       per_tensor={n: {"other": D[n]} for n in BOTTOM})
m = s.measure("H3b bottom conv+L_12 to 55 % parent")
report("H3b", m)
k3b = decide("H3b", m)

# ---------------------------------------------------------------- pass 6: H5
h = s.hypothesis(
    "H5: at the kept H1 state the readable lines have 345 deletions against 148 insertions, and the lines all "
    "three models fail on are read as sparse fragments (mostly blanks). The blank bias is +1.00 (0.7*0.998 + "
    "0.3*0.879 = 0.96 after H1). Lowering it by 0.40 turns low-margin blank frames into characters: fewer "
    "deletions, more insertions.",
    "against the kept state: CER -0.003 (range -0.008 to +0.003); deletions down by 40-80, insertions up by "
    "20-50; collapsed not up; about as likely refused as kept",
    "add on the blank bias")
delta = np.zeros_like(s.subject.tensors()["O_18.lin.bias"])
delta[0] = -0.40
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=delta)
m = s.measure("H5 blank bias -0.40")
report("H5", m)
k5 = decide("H5", m)

print("decisions: H3a", k3a, "H3b", k3b, "H5", k5)
print("budget left:", s.budget_left, "; kept collapsed:", kept_collapsed)
print("blank bias now:", float(s.subject.tensors()["O_18.lin.bias"][0]))
print("digest:", s.subject.digest())
