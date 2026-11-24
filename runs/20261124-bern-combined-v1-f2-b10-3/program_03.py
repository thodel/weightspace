"""program_03 — round 3 of session 20261124-bern-combined-v1-f2-b10-3.

Kept state at start: H1 (30 % toward hertwig-v2, all tensors) + H5 (blank bias -0.40);
CER 0.1359, 20 collapsed.
Pass 7: H4  — a quarter of the subject's share replaced by the sibling: + 0.25 * (sibling - original subject).
Pass 8: H5b — blank bias a further -0.35.
No freeze here.
"""
import json

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

SIBLING = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-schlussredaktion-v1.mlmodel"
ORIGINAL = json.load(open("session.json"))["subject"]      # the unedited subject, read-only

s = open_session(".")
print("budget left at start:", s.budget_left, "; digest", s.subject.digest()[:12])
T = s.subject.tensors()
cm = s.subject.codec_map()
items = list(s.practice)
hand = ["H" if "Hertwig" in it["key"] else "S" for it in items]
kept_collapsed = 20                                         # H5, pass 6 (stdout of program_02)


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


sib = KrakenSubject(SIBLING)
B = sib.tensors()
bcm = sib.codec_map()
rowmap = {cm[ch]: bcm[ch] for ch in cm if ch in bcm}        # subject row -> sibling row, by character
rowmap[0] = 0
print("rows aligned with the sibling:", len(rowmap), "; subject-only:", [ch for ch in cm if ch not in bcm])
orig = KrakenSubject(ORIGINAL)
O = orig.tensors()
assert orig.codec_map() == cm

TRAINED = [n for n in T if "bias_ih" not in n]


def sibling_minus_subject(name):
    """sibling - unedited subject, in the subject's shape; output rows by character, rows the sibling lacks: 0."""
    if not name.startswith("O_18"):
        assert B[name].shape == O[name].shape, name
        return B[name] - O[name]
    d = np.zeros_like(O[name])
    for ia, ib in rowmap.items():
        d[ia] = B[name][ib] - O[name][ia]
    return d


# ---------------------------------------------------------------- pass 7: H4
BETA = 0.25
h = s.hypothesis(
    "H4: subject and sibling are two fine-tunes of the parent on the same Schlussredaktion pages; the sibling "
    "(trained on them with clean data only; 0.113 on the 197 Schlussredaktion practice lines it was trained on, "
    "0.236 on Hertwig, which it was not tuned on) is not much worse than the subject on Hertwig (0.205). "
    "Replacing a quarter of the subject's share by the sibling (state = 0.45 subject + 0.30 parent + 0.25 "
    "sibling, blank bias edit left in place) averages out part of what the subject fitted to mismatched "
    "Hertwig pairs and returns part of the Schlussredaktion cost of H1.",
    "against the kept state: CER -0.0015 (range -0.006 to +0.004): Schlussredaktion -0.004, Hertwig +0.004; "
    "more likely refused than kept; collapsed 20 +- 2",
    "add a donor difference (sibling - subject), all tensors")
s.edit("add", TRAINED, hypothesis=h,
       per_tensor={n: {"delta": (BETA * sibling_minus_subject(n)).astype(np.float32)} for n in TRAINED})
m = s.measure("H4 + 0.25 (sibling - subject)")
report("H4", m)
k4 = decide("H4", m)

# ---------------------------------------------------------------- pass 8: H5b
h = s.hypothesis(
    "H5b: the first blank-bias step (-0.40) removed 86 deletions for 20 insertions and 44 substitutions "
    "(net -22 edits); deletions are still 815 against 189 insertions. A further -0.35 continues with a smaller "
    f"return. (H4 {'kept' if k4 else 'not kept'}.)",
    "against the kept state: CER -0.001 (range -0.004 to +0.002); deletions -40 to -70, insertions and "
    "substitutions together +30 to +70; more likely refused than kept",
    "add on the blank bias")
delta = np.zeros_like(s.subject.tensors()["O_18.lin.bias"])
delta[0] = -0.35
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=delta)
m = s.measure("H5b blank bias further -0.35")
report("H5b", m)
k5b = decide("H5b", m)

print("decisions: H4", k4, "H5b", k5b)
print("budget left:", s.budget_left, "; kept collapsed:", kept_collapsed)
print("blank bias now:", float(s.subject.tensors()["O_18.lin.bias"][0]))
print("digest:", s.subject.digest())
