"""program_02.py — round 1 of session 20261111-bern-combined-v1-f1-b10-1.

Programmer: Claude Fable 5.1 (claude-fable-5-1). Runs inside the sandbox (wsp + numpy only).
Restores the last kept state (H1: blank bias -0.20) and runs, in this order (preregistration.md
section 5, round 1):
    P4 = H1b second blank step (-0.20, total -0.40);
    P5 = H3  space bias +0.30;
    P6 = H6  the '"' output row (weight + bias) from the parent, aligned by character;
    P7 = H4  interpolate alpha 0.25 toward the parent, rows aligned, edited bias rows held.
Each edit is measured, then kept by the harness's rule or reverted; an edit that raises the collapsed
count against the last kept state is reverted even if the rule would keep it.
This program does NOT freeze; round 2 decides the last passes and freezes.
"""
import json
from pathlib import Path

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows
from wsp.metrics import score, align, normalise

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"
PARENT = MODELS + "kraken-bern-hertwig-v2.mlmodel"

s = open_session(".")
s.note("program_02 (round 1), programmer Claude Fable 5.1 (claude-fable-5-1): H1b blank -0.20 more, "
       "H3 space bias +0.30, H6 '\"' row from hertwig-v2, H4 interpolate alpha 0.25 toward hertwig-v2")
print(f"budget {s.budget}, used {s._used}, left {s.budget_left}")

refs = [it["text"] for it in s.practice]
hand = ["H" if it["key"].startswith("A_V_1443") else "S" for it in s.practice]


def last_kept_collapsed():
    """Collapsed count of the last kept state, from log.jsonl (the baseline if nothing was kept)."""
    events = [json.loads(l) for l in Path("log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = None
    last_pass = None
    for e in events:
        if e["kind"] == "pass":
            last_pass = e
            if kept is None:
                kept = e["metrics"]["collapsed"]          # the baseline is the first kept state
        elif e["kind"] == "decision" and e.get("kept") and last_pass is not None:
            kept = last_pass["metrics"]["collapsed"]
    return kept


state = {"kept_collapsed": last_kept_collapsed()}
print(f"last kept state: collapsed {state['kept_collapsed']}, blank bias "
      f"{float(s.subject.tensors()['O_18.lin.bias'][0]):+.4f}")


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
    quote_del = quote_ins = space_del = space_ins = 0
    for h_, r_ in zip(hy, refs):
        for a, b in align(normalise(r_), normalise(h_)):
            if a == b:
                continue
            if b == "":
                dels += 1
                quote_del += a == '"'
                space_del += a == " "
            elif a == "":
                ins += 1
                quote_ins += b == '"'
                space_ins += b == " "
            else:
                subs += 1
    e = s._log[-1]
    vk = e.get("vs_kept") or {}
    vb = e.get("vs_baseline") or {}
    print(f"[{label}] pass {s._used}: CER {m.cer:.4f} ({m.char_edits}/{m.ref_chars})"
          f"  H {per['H'][0]:.4f} ({per['H'][1]}/{per['H'][2]})  S {per['S'][0]:.4f} ({per['S'][1]}/{per['S'][2]})"
          f"  collapsed {m.collapsed} (H {per['H'][3]}, S {per['S'][3]})  length_ratio {m.length_ratio:.3f}"
          f" (H {per['H'][4]:.3f}, S {per['S'][4]:.3f})  WER {m.wer:.4f}")
    print(f"   del {dels} ins {ins} sub {subs}   space del/ins {space_del}/{space_ins}   \" del/ins {quote_del}/{quote_ins}")
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


cs = s.subject.codec_map()
ROW_SPACE = cs[" "]
ROW_QUOTE = cs['"']
held_bias_rows = {0}                     # output-bias rows edited in this session, held in H4

# ---------------------------------------------------------------- P4: H1b second blank step
T = s.subject.tensors()
d = np.zeros_like(T["O_18.lin.bias"])
d[0] = -0.20
h1b = s.hypothesis(
    "H1b second blank step: H1 (-0.20) was kept with a consistent small gain (16 lines better, 7 worse; "
    "deletions 1041 -> 1002, insertions 153 -> 161). Lower O_18.lin.bias[0] by another 0.20 (total -0.40). "
    "No donor weights.",
    "CER 0 to -0.3 points; deletions down by 20-40, insertions up by at most 15; collapsed not up. "
    "If refused: the blank knob is exhausted at -0.20; no third step.",
    "bias shift (output, blank row)")
s.edit("add", ["O_18.lin.bias"], hypothesis=h1b, delta=d)
print(f"H1b: blank bias -> {float(s.subject.tensors()['O_18.lin.bias'][0]):+.4f}")
m4 = s.measure("H1b blank bias -0.20 more (total -0.40)")
report("H1b", m4)
h1b_kept = decide(m4, "H1b second blank step: predicted 0 to -0.3 points")

# ---------------------------------------------------------------- P5: H3 space bias
T = s.subject.tensors()
d = np.zeros_like(T["O_18.lin.bias"])
d[ROW_SPACE] = +0.30
h3 = s.hypothesis(
    "H3 space bias: in the kept state the space is deleted 159 times and inserted 55 times. Raise the "
    "output bias of ' ' (row %d) by 0.30. No donor weights." % ROW_SPACE,
    "CER -0.1 to -0.3 points; space deletions down by 20-40, space insertions up by at most 20, other "
    "confusions unchanged; collapsed not up. If refused: space deletions are word-gap readings the row "
    "cannot fix by bias; no further per-character bias edits.",
    "bias shift (output, one character row)")
s.edit("add", ["O_18.lin.bias"], hypothesis=h3, delta=d)
m5 = s.measure("H3 space bias +0.30")
report("H3", m5)
h3_kept = decide(m5, "H3 space bias +0.30: predicted -0.1 to -0.3 points, space deletions down")
if h3_kept:
    held_bias_rows.add(ROW_SPACE)

# ---------------------------------------------------------------- P6: H6 the '"' row from the parent
par = KrakenSubject(PARENT)
Tp = par.tensors()
cp = par.codec_map()
rq = cp['"']
T = s.subject.tensors()
print(f"H6: '\"' row subject {ROW_QUOTE} (norm {np.linalg.norm(T['O_18.lin.weight'][ROW_QUOTE]):.2f}, "
      f"bias {float(T['O_18.lin.bias'][ROW_QUOTE]):+.3f}) <- parent row {rq} "
      f"(norm {np.linalg.norm(Tp['O_18.lin.weight'][rq]):.2f}, bias {float(Tp['O_18.lin.bias'][rq]):+.3f})")
h6 = s.hypothesis(
    "H6 the '\"' output row from the parent: the subject deletes '\"' 16 of 17 times (Hertwig lines); its "
    "row is the one practice-relevant row with the label-noise signature (norm 3.03 -> 4.27, bias -0.01 -> "
    "-0.15, cosine 0.894 to the parent's). Replace weight row and bias of '\"' by hertwig-v2's, aligned by "
    "character. The parent was trained on the Hertwig practice lines with correct images; this takes its "
    "400-weight classifier for one glyph, not its recurrent stack.",
    "CER -0.05 to -0.15 points, all on Hertwig lines: '\"' deletions fall from 16 to at most 8, at most a "
    "handful of '\"' insertions; Schlussredaktion unchanged; collapsed not up. If refused: the glyph is lost "
    "in the features, not in the row.",
    "set_rows (one output row from the parent, aligned by character)")
s.edit("set_rows", ["O_18.lin.weight", "O_18.lin.bias"], hypothesis=h6, index=[ROW_QUOTE],
       per_tensor={"O_18.lin.weight": {"values": Tp["O_18.lin.weight"][rq][None, :].copy()},
                   "O_18.lin.bias": {"values": np.array([Tp["O_18.lin.bias"][rq]], dtype=np.float32)}})
m6 = s.measure("H6 quote row from hertwig-v2")
report("H6", m6)
h6_kept = decide(m6, "H6 '\"' row from the parent: predicted -0.05 to -0.15 points on Hertwig")

# ---------------------------------------------------------------- P7: H4 toward the parent, alpha 0.25
row_map = codec_row_map(cs, cp)
row_map[0] = 0                                        # blank <-> blank
only_subject = [c for c in cs if c not in cp]
T = s.subject.tensors()
names = [n for n in T if "bias_ih" not in n]          # 28 trained tensors
assert len(names) == 28, names
per_tensor = {}
for n in names:
    if n.startswith("O_18"):
        other = gather_rows(Tp[n], row_map, T[n].shape[0], fill=T[n])
        if n == "O_18.lin.bias":
            for r in sorted(held_bias_rows):
                other[r] = T[n][r]                    # keep this session's bias edits out of the interpolation
        per_tensor[n] = {"other": other}
    else:
        assert Tp[n].shape == T[n].shape, (n, Tp[n].shape, T[n].shape)
        per_tensor[n] = {"other": Tp[n]}
print(f"H4: {len(row_map)} rows aligned; subject-only rows kept: {only_subject}; bias rows held: {sorted(held_bias_rows)}")
h4 = s.hypothesis(
    "H4 toward the parent: interpolate every trained tensor a quarter of the way (alpha 0.25) to "
    "kraken-bern-hertwig-v2, output rows aligned by character, the subject-only rows (½ ƒ ̊) and the bias "
    "rows edited in this session held. The parent was trained on the 103 Hertwig practice lines with correct "
    "images (0.1137 there is a training-set score: a Hertwig gain is partly memorisation) and never saw "
    "Schlussredaktion (0.2992). On Hertwig lines the subject applies Schlussredaktion conventions (,->/ 12, "
    "u->v 11, d->D 8); the parent carries the Hertwig-hand conventions.",
    "Hertwig -1 to -3 points (confounded), Schlussredaktion +0.2 to +1.0 points, net -0.2 to -0.8 points; "
    "collapsed not up. If kept: a practice-set result with a memorisation confound; held-out and C5 decide. "
    "If refused: no larger alpha.",
    "interpolate toward the parent (whole network, rows aligned)")
s.edit("interpolate", names, hypothesis=h4, alpha=0.25, per_tensor=per_tensor)
m7 = s.measure("H4 interpolate alpha 0.25 toward hertwig-v2")
report("H4", m7)
h4_kept = decide(m7, "H4 toward the parent alpha 0.25: predicted net -0.2 to -0.8 points (Hertwig part confounded)")

print(f"program_02 done: H1b kept={h1b_kept}, H3 kept={h3_kept}, H6 kept={h6_kept}, H4 kept={h4_kept}; "
      f"passes left {s.budget_left}. Not frozen.")
