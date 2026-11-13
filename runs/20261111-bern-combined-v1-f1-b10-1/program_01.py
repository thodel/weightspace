"""program_01.py — round 0 of session 20261111-bern-combined-v1-f1-b10-1.

Programmer: Claude Fable 5.1 (claude-fable-5-1). Runs inside the sandbox (wsp + numpy only).

Passes: P1 = C0 baseline (must reproduce 0.1456, else stop before any edit);
        P2 = H1 blank bias -0.20 on O_18.lin.bias[0];
        P3 = H2 soup with the sibling (interpolate, alpha 0.5, output rows aligned by character).
Each edit is measured, then kept by the harness's rule or reverted; an edit that raises the
collapsed count against the last kept state is reverted even if the rule would keep it.
This program does NOT freeze: later rounds decide the follow-ups (preregistration.md section 5).
"""
import json
import sys
from pathlib import Path

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows
from wsp.metrics import score, align, normalise

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"
SIBLING = MODELS + "kraken-bern-schlussredaktion-v1.mlmodel"
C0_CER = 0.1456          # c0_c1.md, the subject on bern-train-300
C0_TOL = 0.0006          # one edit on 10829 characters is 0.0001

s = open_session(".")
s.note("program_01 (round 0), programmer Claude Fable 5.1 (claude-fable-5-1): C0, H1 blank bias -0.20, "
       "H2 soup with schlussredaktion-v1 at alpha 0.5")
print(f"budget {s.budget}, left {s.budget_left}")

refs = [it["text"] for it in s.practice]
hand = ["H" if it["key"].startswith("A_V_1443") else "S" for it in s.practice]
state = {"kept_collapsed": None}


def report(label, m):
    """Print the pass: CER per hand, collapsed, del/ins/sub, confusions and the rule's interval."""
    rd = Path("readings") / f"pass_{s._used:03d}.jsonl"
    hyps = {}
    for line in rd.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            hyps[r["key"]] = r["text"]
    hy = [hyps[it["key"]] for it in s.practice]
    per = {}
    for h in "HS":
        idx = [i for i in range(len(hand)) if hand[i] == h]
        mh = score([hy[i] for i in idx], [refs[i] for i in idx], blocks=False)
        per[h] = (mh.cer, mh.char_edits, mh.ref_chars, mh.collapsed, mh.length_ratio)
    dels = ins = subs = 0
    for h_, r_ in zip(hy, refs):
        for a, b in align(normalise(r_), normalise(h_)):
            if a == b:
                continue
            if b == "":
                dels += 1
            elif a == "":
                ins += 1
            else:
                subs += 1
    e = s._log[-1]
    vk = e.get("vs_kept") or {}
    vb = e.get("vs_baseline") or {}
    print(f"[{label}] pass {s._used}: CER {m.cer:.4f} ({m.char_edits}/{m.ref_chars})"
          f"  H {per['H'][0]:.4f} ({per['H'][1]}/{per['H'][2]})  S {per['S'][0]:.4f} ({per['S'][1]}/{per['S'][2]})"
          f"  collapsed {m.collapsed} (H {per['H'][3]}, S {per['S'][3]})  length_ratio {m.length_ratio:.3f}"
          f" (H {per['H'][4]:.3f}, S {per['S'][4]:.3f})  WER {m.wer:.4f}")
    print(f"   del {dels} ins {ins} sub {subs}")
    print(f"   vs_kept delta {vk.get('delta_cer', float('nan')):+.4f} ci95 {vk.get('ci95')}"
          f"   vs_baseline delta {vb.get('delta_cer', float('nan')):+.4f} ci95 {vb.get('ci95')}")
    print("   confusions:", ", ".join(f"{a or '∅'}→{b or '∅'} {n}" for a, b, n in m.confusions))
    return per


def decide(m, reason):
    """Keep by the rule, unless the collapsed count rose against the last kept state (then revert)."""
    kc = state["kept_collapsed"]
    if kc is not None and m.collapsed > kc:
        s.revert(f"collapsed rose {kc} -> {m.collapsed}; {reason}")
        print(f"   -> REVERTED by the programmer: collapsed rose {kc} -> {m.collapsed}")
        return False
    ok = s.keep(reason)
    if ok:
        state["kept_collapsed"] = m.collapsed
        print("   -> KEPT (rule met)")
    else:
        print(f"   -> REFUSED by the rule: {s._log[-1].get('reason')}")
    return ok


# ---------------------------------------------------------------- P1: C0
m0 = s.measure("C0")
report("C0", m0)
state["kept_collapsed"] = m0.collapsed
if abs(m0.cer - C0_CER) > C0_TOL:
    print(f"STOP: baseline CER {m0.cer:.4f} does not reproduce c0_c1.md ({C0_CER:.4f}); no edit made.")
    sys.exit(1)
if m0.collapsed != 29:
    print(f"note: collapsed {m0.collapsed}, c0_c1.md says 29 (CER reproduced; continuing)")

# ---------------------------------------------------------------- P2: H1 blank bias
T = s.subject.tensors()
b = T["O_18.lin.bias"]
blank_before = float(b[0])
d = np.zeros_like(b)
d[0] = -0.20
h1 = s.hypothesis(
    "H1 blank bias: the subject's CTC blank bias is +1.00, 0.12 above the parent's and 0.06 above the "
    "sibling's, and its errors are deletions (a-z 669 del vs 48 ins; space 168 vs 54; '/' 70 vs 37). "
    "Lower O_18.lin.bias[0] by 0.20. No donor weights are used.",
    "CER down 0.1 to 0.5 points on both hands; deletions of space, e, n, i, r, t fall, insertions rise by "
    "less; collapsed count does not rise. If refused: the deletions sit in the features or in unreadable "
    "lines, not at the blank threshold; no further blank steps.",
    "bias shift (output, blank row)")
s.edit("add", ["O_18.lin.bias"], hypothesis=h1, delta=d)
print(f"H1: blank bias {blank_before:+.4f} -> {float(s.subject.tensors()['O_18.lin.bias'][0]):+.4f}")
m1 = s.measure("H1 blank bias -0.20")
report("H1", m1)
h1_kept = decide(m1, "H1 blank bias -0.20: predicted -0.1 to -0.5 points, deletions down")

# ---------------------------------------------------------------- P3: H2 soup with the sibling
sib = KrakenSubject(SIBLING)
Tb = sib.tensors()
cs = s.subject.codec_map()
cb = sib.codec_map()
row_map = codec_row_map(cs, cb)
row_map[0] = 0                                   # blank <-> blank
only_subject = [c for c in cs if c not in cb]
print(f"H2: {len(row_map)} output rows aligned by character; subject-only rows kept: {only_subject}")
T = s.subject.tensors()
names = [n for n in T if "bias_ih" not in n]     # 28 trained tensors; bias_ih is zero everywhere
assert len(names) == 28, names
per_tensor = {}
for n in names:
    if n.startswith("O_18"):
        other = gather_rows(Tb[n], row_map, T[n].shape[0], fill=T[n])
        if n == "O_18.lin.bias" and h1_kept:
            other[0] = T[n][0]                   # hold H1's blank bias; the blank row weight is still soup
        per_tensor[n] = {"other": other}
    else:
        assert Tb[n].shape == T[n].shape, (n, Tb[n].shape, T[n].shape)
        per_tensor[n] = {"other": Tb[n]}
h2 = s.hypothesis(
    "H2 soup with the sibling: interpolate every trained tensor halfway (alpha 0.5) to "
    "kraken-bern-schlussredaktion-v1, output rows aligned by character, the subject-only row 'ƒ' unchanged"
    + (", H1's blank bias held" if h1_kept else "") + ". The sibling has the same parent and was trained on "
    "the clean Schlussredaktion training pages, i.e. on the 197 Schlussredaktion practice lines (as the "
    "subject was, with correct images: equal footing there); it never saw Hertwig, so a Hertwig gain from "
    "this edit cannot be memorisation. The two fine-tune directions have cosine 0.52 (0.85-0.97 in O_18, "
    "0.24-0.28 in L_12).",
    "Schlussredaktion down 0.3 to 0.8 points (toward or beyond the sibling's 0.1129); Hertwig between -1 "
    "and +1 point; overall -0.2 to -0.6 points; collapsed unchanged. If refused with both hands up: not "
    "linearly connected at 0.5, try alpha 0.25. If refused with S down and H up: the Hertwig knowledge "
    "lies off the sibling's direction (H4).",
    "interpolate toward a sibling (whole network, rows aligned)")
s.edit("interpolate", names, hypothesis=h2, alpha=0.5, per_tensor=per_tensor)
m2 = s.measure("H2 soup sibling alpha 0.5")
report("H2", m2)
h2_kept = decide(m2, "H2 soup with schlussredaktion-v1 at alpha 0.5: predicted -0.2 to -0.6 points, S down")

print(f"program_01 done: H1 kept={h1_kept}, H2 kept={h2_kept}; passes left {s.budget_left}. Not frozen.")
