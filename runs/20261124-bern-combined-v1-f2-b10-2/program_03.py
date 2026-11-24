"""program_03 — round 3 and last program of 20261124-bern-combined-v1-f2-b10-2.

Kept state on entry: 0.75 subject + 0.25 parent in every tensor (H2), blank bias -0.7 (H3).
  M1 (pass 7): one edit with two parts: late block (L_16, O_18) back to the unedited subject (blank bias
               untouched) AND conv stack, L_12, L_14 further toward the parent (to 0.45 parent).
  M2 (pass 8): only if M1 is kept: conv stack, L_12, L_14 to 0.60 parent.
Then freeze. Passes 9-10 (and 8, if M1 is not kept) stay unspent.
"""
import json
import sys
from pathlib import Path

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.metrics import per_line_edits, bootstrap_delta

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"

s = open_session(".")
print("budget_left at start:", s.budget_left)
if s.budget_left != 4:
    print("STOP: program_03 expects 4 passes left; nothing done, not frozen.")
    sys.exit(0)

T = s.subject.tensors()
cm = s.subject.codec_map()
refs = [it["text"] for it in s.practice]
is_h = [("Hertwig" in it["key"]) for it in s.practice]
ALL = [n for n in T if "bias_ih" not in n]
FRONTMID = [n for n in ALL if n.startswith(("C_0.", "C_3.", "C_6.", "C_9.", "L_12.", "L_14."))]
LATE = [n for n in ALL if n.startswith(("L_16.", "O_18."))]
assert len(FRONTMID) == 20 and len(LATE) == 8, (len(FRONTMID), len(LATE))


def read_lines(n):
    rows = [json.loads(l) for l in Path(f"readings/pass_{n:03d}.jsonl").read_text().splitlines() if l.strip()]
    assert [r["key"] for r in rows] == [it["key"] for it in s.practice]
    return per_line_edits([r["text"] for r in rows], refs)


def cer(lines, flag=None):
    sel = [l for l, h in zip(lines, is_h) if flag is None or h == flag]
    return sum(e for e, _ in sel) / sum(c for _, c in sel)


def hand_delta(before, after):
    out = {}
    for name, flag in (("Hertwig", True), ("Schlussred", False)):
        b = [l for l, h in zip(before, is_h) if h == flag]
        a = [l for l, h in zip(after, is_h) if h == flag]
        d = bootstrap_delta(b, a, resamples=2000, seed=20261106)
        out[name] = (round(d["delta_cer"], 4), [round(x, 4) for x in d["ci95"]])
    return out


lines_c0 = read_lines(1)
bad0 = [c > 0 and e > 0.5 * c for e, c in lines_c0]


def groups(lines):
    out = {}
    for name, f in (("failed@C0 H", lambda i: bad0[i] and is_h[i]), ("failed@C0 S", lambda i: bad0[i] and not is_h[i]),
                    ("ok@C0 H", lambda i: not bad0[i] and is_h[i]), ("ok@C0 S", lambda i: not bad0[i] and not is_h[i])):
        out[name] = sum(lines[i][0] for i in range(len(lines)) if f(i))
    return out


kept_lines = read_lines(4)                               # pass 4 (H3) is still the last kept state
kept_collapsed = 18
print(f"kept state on entry: CER {cer(kept_lines):.4f} (H {cer(kept_lines, True):.4f}, S {cer(kept_lines, False):.4f}); "
      f"edits by group {groups(kept_lines)}; blank bias {float(T['O_18.lin.bias'][0]):+.4f}")

orig = KrakenSubject(MODELS + "kraken-bern-combined-v1.mlmodel")
assert orig.codec_map() == cm
O = {n: orig.tensors()[n].copy() for n in ALL}
del orig
par = KrakenSubject(MODELS + "kraken-bern-hertwig-v2.mlmodel")
P = {n: par.tensors()[n].copy() for n in FRONTMID}       # no output layer among them: no row alignment needed
del par
for n in FRONTMID:
    assert P[n].shape == T[n].shape, n

# the restored state must be the kept one: 0.75 subject + 0.25 parent in the front, H3 blank bias
chk = "L_12.layer.weight_ih_l0"
dev = float(np.abs(T[chk] - (0.75 * O[chk] + 0.25 * P[chk])).max())
print(f"state check: max |T - (0.75 subject + 0.25 parent)| on {chk} = {dev:.2e}")
if abs(cer(kept_lines) - 0.1346) > 5e-4 or abs(float(T["O_18.lin.bias"][0]) - 0.268) > 0.01 or dev > 1e-4:
    print("STOP: the restored state is not the kept state of round 1; nothing done, not frozen.")
    sys.exit(0)


def report(label, m, lines):
    print(f"\n=== {label} ===")
    print(f"CER {m.cer:.4f}  edits {m.char_edits}/{m.ref_chars}  collapsed {m.collapsed}  "
          f"length_ratio {m.length_ratio:.3f}  WER {m.wer:.4f}")
    print(f"per hand: Hertwig {cer(lines, True):.4f}  Schlussred {cer(lines, False):.4f}")
    print("edits by group:", groups(lines), "| kept state:", groups(kept_lines))
    print("confusions:", m.confusions)
    e = s._log[-1]
    vk, vb = e["vs_kept"], e["vs_baseline"]
    print(f"vs_kept delta {vk['delta_cer']:+.4f} ci95 [{vk['ci95'][0]:+.4f}, {vk['ci95'][1]:+.4f}]   "
          f"vs_baseline delta {vb['delta_cer']:+.4f} ci95 [{vb['ci95'][0]:+.4f}, {vb['ci95'][1]:+.4f}]")
    print("per-hand delta vs kept (delta, ci95):", hand_delta(kept_lines, lines))
    print("lines better", sum(1 for a, b in zip(lines, kept_lines) if a[0] < b[0]),
          "worse", sum(1 for a, b in zip(lines, kept_lines) if a[0] > b[0]))


def decide(label, m, lines):
    global kept_lines, kept_collapsed
    if m.collapsed > kept_collapsed:
        s.revert(f"{label}: collapsed rose {kept_collapsed} -> {m.collapsed}")
        print(f"DECISION {label}: reverted (collapsed rose {kept_collapsed} -> {m.collapsed})")
        return False
    kept = s.keep(label)
    print(f"DECISION {label}: {'KEPT' if kept else 'refused by the keep rule, undone'}")
    if kept:
        kept_lines, kept_collapsed = lines, m.collapsed
    return kept


def with_current_blank(name, arr):
    if name == "O_18.lin.bias":
        arr = arr.copy()
        arr[0] = T[name][0]
    return arr


# ---------------------------------------------------------------- M1, pass 7
A1 = 1.0 - 0.55 / 0.75            # from 0.75 subject + 0.25 parent to 0.55 subject + 0.45 parent
h = s.hypothesis(
    "M1: passes 3-6 on the lines readable at C0 say that the parent share pays in the front and middle "
    "(conv stack, L_12, L_14 at 0.25 parent with the late block at the subject: Hertwig -73 edits, Schlussredaktion "
    "+23 against C0) and does not pay in the late block (adding it there: Hertwig -27, Schlussredaktion +57). "
    "So one edit with two parts: L_16 and O_18 back to the unedited subject's values (blank bias left at its kept "
    "value), and conv stack, L_12 and L_14 from 0.25 to 0.45 parent. The first part alone was pass 5 (-0.17 points, "
    "refused). The parent was trained on all 103 Hertwig practice lines, so the Hertwig part of the gain may be "
    "recall; the Schlussredaktion part is on a hand the parent never saw.",
    "practice CER -0.3 points against the kept state (range -0.9 to +0.4); Schlussredaktion -0.2 to -0.7, "
    "Hertwig -0.5 to +1.0; collapsed 18 +/- 2; probability that the rule keeps it about 0.35",
    "block-wise mix with the parent: interpolate, late block alpha 1 toward the original subject, "
    "front and middle alpha 0.267 toward the parent")
pt = {n: {"other": with_current_blank(n, O[n]), "alpha": 1.0} for n in LATE}
pt.update({n: {"other": P[n], "alpha": A1} for n in FRONTMID})
s.edit("interpolate", LATE + FRONTMID, hypothesis=h, per_tensor=pt)
m = s.measure("M1 late=subject, front+mid 0.45 parent")
lines = read_lines(7)
report("M1 late block = subject, front+middle 0.45 parent (pass 7)", m, lines)
k1 = decide("M1 late=subject, front+mid 0.45 parent", m, lines)

# ---------------------------------------------------------------- M2, pass 8 (only if M1 is kept)
k2 = None
if k1:
    A2 = 1.0 - 0.40 / 0.55        # from 0.45 to 0.60 parent
    h = s.hypothesis(
        "M2: M1 was kept, so a larger parent share in the front and middle still pays with the late block at the "
        "subject. One more step of conv stack, L_12 and L_14, from 0.45 to 0.60 parent. I expect the gain to "
        "flatten here: the Schlussredaktion cost grows faster than the Hertwig gain along this line.",
        "practice CER -0.1 points against the kept state (range -0.6 to +0.6); Hertwig -0.5 to -1.5, "
        "Schlussredaktion +0.2 to +0.8; probability that the rule keeps it about 0.2",
        "interpolate toward donor (parent), blocks C_0..C_9, L_12, L_14, alpha 0.273")
    s.edit("interpolate", FRONTMID, hypothesis=h, per_tensor={n: {"other": P[n]} for n in FRONTMID}, alpha=A2)
    m = s.measure("M2 front+mid 0.60 parent")
    lines = read_lines(8)
    report("M2 front+middle 0.60 parent (pass 8)", m, lines)
    k2 = decide("M2 front+mid 0.60 parent", m, lines)
else:
    s.note("M2 not run: it was conditional on M1 being kept")
    print("\nM2 not run (M1 not kept).")

# ---------------------------------------------------------------- freeze
front_share = float(np.abs(T[chk] - O[chk]).sum() / np.abs(P[chk] - O[chk]).sum())
late_changed = bool(np.abs(T["L_16.layer.weight_ih_l0"] - O["L_16.layer.weight_ih_l0"]).max() > 0)
print(f"\nfinal state: M1 {k1}, M2 {k2}; parent share in the front ~{front_share:.2f}; late block differs from subject: "
      f"{late_changed}; blank bias {float(T['O_18.lin.bias'][0]):+.4f}; "
      f"kept CER {cer(kept_lines):.4f} (H {cer(kept_lines, True):.4f}, S {cer(kept_lines, False):.4f}); "
      f"collapsed {kept_collapsed}; passes unspent {s.budget_left}")
s.note(f"freeze: kept edits H2 (parent a=0.25), H3 (blank bias -0.7), M1 {k1}, M2 {k2}; {s.budget_left} passes unspent")
summary = s.freeze()
print("frozen. used", summary["used"], "of", summary["budget"], "| edits", summary["edits"], "| kept", summary["kept"],
      "| final digest", summary["final_digest"][:16])
print("changed tensors:", len(summary["changed_tensors"]))
