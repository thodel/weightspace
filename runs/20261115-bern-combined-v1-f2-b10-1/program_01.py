"""program_01: pass 1 = C0, then H1 (average with the sibling), H2 (quarter step toward the
parent), H3 (blank bias -0.3). Each edit: hypothesis -> edit -> measure -> keep / revert.
Runs inside the sandbox; only the wsp API and numpy. No freeze here."""
import sys

import numpy as np

from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"
PARENT = MODELS + "kraken-bern-hertwig-v2.mlmodel"
SIBLING = MODELS + "kraken-bern-schlussredaktion-v1.mlmodel"
OUT_W, OUT_B = "O_18.lin.weight", "O_18.lin.bias"

s = open_session(".")
print("budget left at start:", s.budget_left)
cm = s.subject.codec_map()
hertwig = ["Hertwig" in it["key"] for it in s.practice]


def per_hand():
    """CER per hand from the per-line edits of the last pass."""
    out = {}
    for name, flag in (("Hertwig", True), ("Schlussred", False)):
        rows = [r for r, h in zip(s._last_lines, hertwig) if h == flag]
        out[name] = sum(e for e, _ in rows) / sum(c for _, c in rows)
    return out


def report(label, m):
    e = s._log[-1]
    ph = per_hand()
    print(f"\n== {label}: CER {m.cer:.4f} ({m.char_edits}/{m.ref_chars}) "
          f"Hertwig {ph['Hertwig']:.4f} Schlussred {ph['Schlussred']:.4f} "
          f"collapsed {m.collapsed} length_ratio {m.length_ratio:.4f} WER {m.wer:.4f}")
    print("   vs_kept:", {k: e["vs_kept"][k] for k in ("delta_cer", "ci95")})
    print("   vs_baseline:", {k: e["vs_baseline"][k] for k in ("delta_cer", "ci95")})
    print("   confusions:", m.confusions)
    print("   blocks:", m.blocks)
    return ph


def donor_arrays(donor):
    """Donor tensors in the subject's shapes. Output rows are aligned by character; a row the
    donor lacks takes the subject's current values, so an interpolation leaves it unchanged."""
    T, D, dcm = s.subject.tensors(), donor.tensors(), donor.codec_map()
    names, arrays = [], {}
    for n, w in T.items():
        if n in (OUT_W, OUT_B):
            a = w.copy()
            a[0] = D[n][0]                       # the CTC blank is row 0 in both
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


def decide(label, m, kept_collapsed, reason):
    """Keep by the harness rule; revert first if the collapsed count rose."""
    if m.collapsed > kept_collapsed:
        s.revert(f"{label}: collapsed rose {kept_collapsed} -> {m.collapsed}")
        print(f"   DECISION {label}: reverted (collapsed rose {kept_collapsed} -> {m.collapsed})")
        return False
    kept = s.keep(reason)
    print(f"   DECISION {label}: {'KEPT' if kept else 'refused by the keep rule, undone'}")
    return kept


# ---- pass 1: C0 -------------------------------------------------------------------------
m0 = s.measure("C0")
report("C0", m0)
if abs(m0.cer - 0.1456) > 0.00006 or m0.char_edits != 1577:
    print(f"STOP: C0 does not reproduce c0_c1.md (expected CER 0.1456, 1577 edits; got "
          f"{m0.cer:.6f}, {m0.char_edits}). No further pass.")
    s.note("C0 did not reproduce the pack's line; program_01 stopped after pass 1")
    sys.exit(0)
kept_collapsed = m0.collapsed
state = []

# ---- pass 2: H1, average with the clean sibling ------------------------------------------
sib = KrakenSubject(SIBLING)
names, arrays = donor_arrays(sib)
h = s.hypothesis(
    "H1: interpolate all tensors halfway toward schlussredaktion-v1 (sibling, same parent, trained on "
    "the clean Schlussredaktion pages = the 197 Schlussredaktion practice lines). Output rows by "
    "character; the row of 'ƒ' stays. Averaging halves the part of the subject's delta the sibling "
    "does not share, which includes what the mismatched Hertwig pairs taught.",
    "total -0.3 to -1.0 pt; Schlussredaktion -0.3 to -1.0 pt; Hertwig -1 to +1.5 pt; collapsed not up; "
    "keep chance about 40 %",
    "donor interpolation, all tensors")
s.edit("interpolate", names, hypothesis=h, alpha=0.5, per_tensor=arrays)
m = s.measure("H1 sibling alpha=0.5 all tensors")
report("H1", m)
if decide("H1", m, kept_collapsed, "H1: average with the sibling lowers practice CER"):
    kept_collapsed = m.collapsed
    state.append("H1")
del sib

# ---- pass 3: H2, a quarter step toward the parent ----------------------------------------
par = KrakenSubject(PARENT)
names, arrays = donor_arrays(par)
h = s.hypothesis(
    "H2: interpolate all tensors a quarter of the way toward hertwig-v2 (parent, trained on the Hertwig "
    "training pages with correct images = all 103 Hertwig practice lines; reads Schlussredaktion at "
    f"0.299). Output rows by character; the rows of '½ ƒ ̊' stay. Applied on the kept state {state or 'C0'}.",
    "Hertwig -1.5 to -4 pt; Schlussredaktion +0.5 to +2.5 pt; total between -1.0 and +1.0 pt, sign open; "
    "keep chance about 30 %; a Hertwig gain is on lines the parent was trained on",
    "donor interpolation, all tensors")
s.edit("interpolate", names, hypothesis=h, alpha=0.25, per_tensor=arrays)
m = s.measure("H2 parent alpha=0.25 all tensors")
report("H2", m)
if decide("H2", m, kept_collapsed, "H2: quarter step toward the parent lowers practice CER"):
    kept_collapsed = m.collapsed
    state.append("H2")
del par

# ---- pass 4: H3, blank bias down ---------------------------------------------------------
bias = s.subject.tensors()[OUT_B]
delta = np.zeros_like(bias)
delta[0] = -0.3
h = s.hypothesis(
    f"H3: lower the blank bias O_18.lin.bias[0] by 0.3 (now {float(bias[0]):+.3f}) on the kept state "
    f"{state or 'C0'}. Deletions dominate the subject's errors and the blank has the highest bias.",
    "total -0.2 to -0.6 pt; deletions down 40-100, insertions up 20-60; length ratio +0.005 to +0.015; "
    "collapsed unchanged; keep chance about 45 %",
    "output bias, blank row")
s.edit("add", [OUT_B], hypothesis=h, delta=delta)
m = s.measure("H3 blank bias -0.3")
report("H3", m)
if decide("H3", m, kept_collapsed, "H3: lower blank bias lowers practice CER"):
    kept_collapsed = m.collapsed
    state.append("H3")

print("\nkept after program_01:", state or "nothing", "| collapsed of kept state:", kept_collapsed,
      "| budget left:", s.budget_left)
