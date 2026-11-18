"""program_02: passes 5-8 on the kept state (0.75 subject + 0.25 parent, blank bias -0.3).
H4 parent step in the front blocks, H5 parent step in the back blocks, H6 step toward the
sibling, H7 second blank dose. Each edit: hypothesis -> edit -> measure -> keep / revert.
Runs inside the sandbox; only the wsp API and numpy. No freeze here."""
import json
import sys

import numpy as np

from wsp.kraken_subject import KrakenSubject
from wsp.metrics import is_collapsed
from wsp.run import open_session

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"
PARENT = MODELS + "kraken-bern-hertwig-v2.mlmodel"
SIBLING = MODELS + "kraken-bern-schlussredaktion-v1.mlmodel"
OUT_W, OUT_B = "O_18.lin.weight", "O_18.lin.bias"
FRONT = ("C_0.", "C_3.", "C_6.", "C_9.", "L_12.")
BACK = ("L_14.", "L_16.", "O_18.")

s = open_session(".")
print("budget left at start:", s.budget_left)
if s.budget_left != 6:
    print("STOP: expected 6 passes left")
    sys.exit(0)
cm = s.subject.codec_map()
hertwig = ["Hertwig" in it["key"] for it in s.practice]

# collapsed count of the kept state = the last kept pass (pass 4, H3), from its readings
hyp4 = [json.loads(l)["text"] for l in open("readings/pass_004.jsonl")]
kept_collapsed = sum(is_collapsed(h, it["text"]) for h, it in zip(hyp4, s.practice))
print("collapsed of the kept state (pass 4):", kept_collapsed, "| blank bias now:",
      float(s.subject.tensors()[OUT_B][0]))
if kept_collapsed != 21:
    print("STOP: the kept state's collapsed count is not the 21 of program_01")
    sys.exit(0)


def per_hand():
    out = {}
    for name, flag in (("Hertwig", True), ("Schlussred", False)):
        rows = [r for r, h in zip(s._last_lines, hertwig) if h == flag]
        out[name] = sum(e for e, _ in rows) / sum(c for _, c in rows)
    return out


def report(label, m):
    e = s._log[-1]
    ph = per_hand()
    b = m.blocks.get("blocks", {}) if isinstance(m.blocks, dict) else {}
    tot = {k: sum(v.get(k, 0) for v in b.values() if not v.get("overlay")) for k in ("del", "ins", "sub")}
    print(f"\n== {label}: CER {m.cer:.4f} ({m.char_edits}/{m.ref_chars}) "
          f"Hertwig {ph['Hertwig']:.4f} Schlussred {ph['Schlussred']:.4f} "
          f"collapsed {m.collapsed} length_ratio {m.length_ratio:.4f} WER {m.wer:.4f}")
    print("   del/ins/sub:", tot)
    print("   vs_kept:", {k: e["vs_kept"][k] for k in ("delta_cer", "ci95")})
    print("   vs_baseline:", {k: e["vs_baseline"][k] for k in ("delta_cer", "ci95")})
    print("   confusions:", m.confusions)


def donor_arrays(donor, prefixes):
    """Donor tensors in the subject's shapes for the blocks named by ``prefixes``. Output rows
    are aligned by character; a row the donor lacks keeps the subject's current values, and the
    blank bias keeps its current value (the blank edits stay as they are)."""
    T, D, dcm = s.subject.tensors(), donor.tensors(), donor.codec_map()
    names, arrays = [], {}
    for n, w in T.items():
        if not n.startswith(prefixes):
            continue
        if n in (OUT_W, OUT_B):
            a = w.copy()
            if n == OUT_W:
                a[0] = D[n][0]                   # blank weight row; the blank bias is left alone
            for ch, i in cm.items():
                if ch in dcm:
                    a[i] = D[n][dcm[ch]]
        elif n in D and D[n].shape == w.shape:
            a = D[n]
        else:
            raise RuntimeError(f"no donor tensor for {n}")
        if not np.array_equal(a, w):             # the all-zero bias_ih tensors are skipped
            names.append(n)
            arrays[n] = {"other": a.astype(np.float32)}
    return names, arrays


def decide(label, m, reason):
    """Keep by the harness rule; revert first if the collapsed count rose."""
    global kept_collapsed
    if m.collapsed > kept_collapsed:
        s.revert(f"{label}: collapsed rose {kept_collapsed} -> {m.collapsed}")
        print(f"   DECISION {label}: reverted (collapsed rose {kept_collapsed} -> {m.collapsed})")
        return False
    kept = s.keep(reason)
    print(f"   DECISION {label}: {'KEPT' if kept else 'refused by the keep rule, undone'}")
    if kept:
        kept_collapsed = m.collapsed
        state.append(label)
    return kept


state = ["H2", "H3"]
par = KrakenSubject(PARENT)

# ---- pass 5: H4, parent step in the front blocks ------------------------------------------
names, arrays = donor_arrays(par, FRONT)
print("H4 tensors:", len(names))
h = s.hypothesis(
    "H4: interpolate C_0, C_3, C_6, C_9 and L_12 a further quarter toward hertwig-v2 (parent; trained "
    "on all 103 Hertwig practice lines with correct images, never on Schlussredaktion). Parent share "
    "there 0.25 -> 0.44. Asks whether the Hertwig gain and the Schlussredaktion cost of H2 sit in "
    "different blocks.",
    "Hertwig -0.5 to -2 pt; Schlussredaktion +0.3 to +1.5 pt; total between -0.4 and +0.4 pt; "
    "keep chance about 25 %",
    "donor interpolation, front blocks")
s.edit("interpolate", names, hypothesis=h, alpha=0.25, per_tensor=arrays)
m = s.measure("H4 parent alpha=0.25 front (C_*, L_12)")
report("H4", m)
decide("H4", m, "H4: further parent step in the front blocks lowers practice CER")

# ---- pass 6: H5, parent step in the back blocks -------------------------------------------
names, arrays = donor_arrays(par, BACK)
print("H5 tensors:", len(names))
h = s.hypothesis(
    "H5: interpolate L_14, L_16 and O_18 a further quarter toward hertwig-v2 (parent, as in H4). Output "
    f"rows by character; the blank bias and the rows of '½ ƒ ̊' stay. On the kept state {state}.",
    "Hertwig -0.5 to -2 pt; Schlussredaktion +0.3 to +1.5 pt; total between -0.4 and +0.4 pt; "
    "keep chance about 25 %",
    "donor interpolation, back blocks")
s.edit("interpolate", names, hypothesis=h, alpha=0.25, per_tensor=arrays)
m = s.measure("H5 parent alpha=0.25 back (L_14, L_16, O_18)")
report("H5", m)
decide("H5", m, "H5: further parent step in the back blocks lowers practice CER")
del par

# ---- pass 7: H6, step toward the sibling ---------------------------------------------------
sib = KrakenSubject(SIBLING)
names, arrays = donor_arrays(sib, FRONT + BACK)
print("H6 tensors:", len(names))
h = s.hypothesis(
    "H6: interpolate all tensors a fifth of the way toward schlussredaktion-v1 (sibling; trained on the "
    "clean Schlussredaktion pages = all 197 Schlussredaktion practice lines, never on Hertwig after the "
    f"parent). Rows by character; the blank bias and the row of 'ƒ' stay. On the kept state {state}. "
    "Asks whether the sibling buys back the Schlussredaktion cost of the parent step.",
    "Schlussredaktion -0.3 to -0.8 pt; Hertwig +0.3 to +1.0 pt; total between -0.3 and +0.2 pt; "
    "keep chance about 25 %",
    "donor interpolation, all tensors")
s.edit("interpolate", names, hypothesis=h, alpha=0.2, per_tensor=arrays)
m = s.measure("H6 sibling alpha=0.2 all tensors")
report("H6", m)
decide("H6", m, "H6: step toward the sibling lowers practice CER")
del sib

# ---- pass 8: H7, second blank dose ---------------------------------------------------------
bias = s.subject.tensors()[OUT_B]
delta = np.zeros_like(bias)
delta[0] = -0.3
h = s.hypothesis(
    f"H7: lower the blank bias by a further 0.3 (now {float(bias[0]):+.3f}) on the kept state {state}. "
    "Deletions still outnumber insertions on the readable lines after H3.",
    "total -0.05 to -0.2 pt; deletions down 15-40, insertions up 10-30; collapsed unchanged; "
    "keep chance about 40 %",
    "output bias, blank row")
s.edit("add", [OUT_B], hypothesis=h, delta=delta)
m = s.measure("H7 blank bias -0.3 more")
report("H7", m)
decide("H7", m, "H7: second blank dose lowers practice CER")

print("\nkept after program_02:", state, "| collapsed of kept state:", kept_collapsed,
      "| budget left:", s.budget_left)
