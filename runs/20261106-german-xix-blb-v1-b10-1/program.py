"""The programmer's script for session 20261106-german-xix-blb-v1-b10-1.

Runs inside the sandbox (env -i, unshare -Un, python -I); only the wsp API and numpy.
Preregistered hypotheses are in preregistration.md; this script tests them in order
and decides by the protocol rule: keep only if practice CER falls and the collapsed
count does not rise; ties revert. Budget 10 passes, pass 1 is the C0 baseline.
"""
import json
import numpy as np
from wsp.run import open_session
from wsp.align import codec_row_map

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
BLANK_DELTA_PARENT = -0.113   # child blank bias 1.2323 - parent 1.1195 (docs/READINGS.md)

s = open_session(".")
print(s.subject.describe()["parameters"], "parameters;", s.budget_left, "passes left")
bias = "O_18.lin.bias"
weight = "O_18.lin.weight"
cm = s.subject.codec_map()
space = cm[" "]


def decide(before, after, what):
    """The protocol's rule, mechanically."""
    ok = after.cer < before.cer and after.collapsed <= before.collapsed
    msg = f"{what}: CER {before.cer:.4f} -> {after.cer:.4f}, collapsed {before.collapsed} -> {after.collapsed}"
    if ok:
        s.keep(msg); print("KEEP", msg)
    else:
        s.revert(msg); print("REVERT", msg)
    return ok


# pass 1: C0 baseline
base = s.measure("C0 baseline")
print("baseline", base.cer, base.collapsed, base.confusions[:5])
best = base

# H1: blank bias back to the parent's value (the readings' first hypothesis)
h1 = s.hypothesis("fine-tuning raised the blank bias by +0.113; putting it back to the parent's value reduces deletions",
                  "CER falls by 0.1-0.5 points (deletions ` e n -> blank` fall, insertions rise less); collapsed lines do not rise",
                  "output bias (blank row)")
d = np.zeros_like(s.subject.tensors()[bias]); d[0] = BLANK_DELTA_PARENT
s.edit("add", [bias], hypothesis=h1, delta=d)
m = s.measure("H1 blank bias -0.113")
if decide(best, m, "H1 blank -0.113"):
    best = m

# H2: a larger step in the same direction (does the direction hold at a scale the bias readings cannot predict?)
h2 = s.hypothesis("if the blank bias is the lever, a larger step (-0.5) lowers CER further than -0.113",
                  "CER falls by 0.3-1.0 point against the current state; if the blank is already well placed, CER rises and collapses appear",
                  "output bias (blank row)")
d = np.zeros_like(s.subject.tensors()[bias]); d[0] = -0.5 - (BLANK_DELTA_PARENT if best is not base else 0.0)
s.edit("add", [bias], hypothesis=h2, delta=d)
m = s.measure("H2 blank bias to -0.5 total")
if decide(best, m, "H2 blank -0.5 total"):
    best = m

# H3: the space row. ` ` -> blank is the top confusion (103 of 831 edits at C0) and the space row carries the
# most negative bias of the whole layer (-7.85).
h3 = s.hypothesis("the space row's bias (-7.85, the lowest of the layer) is why spaces are deleted; raising it by +1.0 recovers them",
                  "CER falls by 0.2-0.6 points; ` `->blank count falls by a third; insertions of spaces rise less; collapsed unchanged",
                  "output bias (space row)")
d = np.zeros_like(s.subject.tensors()[bias]); d[space] = 1.0
s.edit("add", [bias], hypothesis=h3, delta=d)
m = s.measure("H3 space bias +1.0")
h3_ok = decide(best, m, "H3 space +1.0")
if h3_ok:
    best = m

# H4: space row, larger step (+2.0 total) if +1.0 helped, else +0.5
step = 1.0 if h3_ok else 0.5
h4 = s.hypothesis(f"space bias: {'a further +1.0 (to +2.0)' if h3_ok else 'a smaller step (+0.5)'} is the better size",
                  "CER falls a further 0.1-0.3 points if the first step undershot; rises if it overshot",
                  "output bias (space row)")
d = np.zeros_like(s.subject.tensors()[bias]); d[space] = step
s.edit("add", [bias], hypothesis=h4, delta=d)
m = s.measure(f"H4 space bias +{step}")
if decide(best, m, f"H4 space +{step}"):
    best = m

# H5: the rare-row reading. The 12 rows that moved most in the fine-tune (`« ¬ _ » … ͇ ̅ ° ￼ ‘ † [`) are typographic
# characters the BLB GT does not use; the readings predict reverting them to the parent changes nothing on practice.
# A null prediction is a prediction: this pass tests the reading, and the edit is reverted unless CER falls.
from wsp.kraken_subject import KrakenSubject
parent = KrakenSubject(PARENT)
pm = parent.codec_map()
moved = ["«", "¬", "_", "»", "…", "͇", "̅", "°", "￼", "‘", "†", "["]
rows = [cm[c] for c in moved if c in cm and c in pm]
prow = [pm[c] for c in moved if c in cm and c in pm]
h5 = s.hypothesis("the 12 output rows that moved most are rare typographic characters; reverting them to the parent is inert on practice text",
                  "|CER change| < 0.05 points, 0-2 lines differ; collapsed unchanged (prediction of docs/READINGS.md §1)",
                  "output rows (set_rows from parent, aligned by character)")
W = s.subject.tensors()[weight]; PW = parent.tensors()[weight]
s.edit("set_rows", [weight], hypothesis=h5, index=rows, values=PW[prow].astype(W.dtype))
m = s.measure("H5 rare rows from parent")
# a null result is the expected outcome; by the rule a tie reverts
decide(best, m, "H5 rare rows")
s.note(f"H5 delta CER {m.cer - best.cer:+.5f} on {len(rows)} rows; readings predicted ~0")

# H6: the frequent rows. The weakest non-blank rows are `e r s t i n a` (norm 7.6-8.3) and they are the deleted
# characters (e->blank 40, n->blank 35). Scaling those rows' weights by 1.1 raises their logits where the input
# already points at them, competing with the blank at the frame level.
freq = [cm[c] for c in "ersitna" if c in cm]
h6 = s.hypothesis("the frequent letters' rows are the weakest and lose frames to the blank; scaling them by 1.1 recovers deletions",
                  "CER falls by 0.2-0.5 points, e/n/i/r/s->blank fall; substitutions among them may rise; collapsed unchanged",
                  "output rows (scale rows of frequent letters)")
Wc = s.subject.tensors()[weight]
s.edit("set_rows", [weight], hypothesis=h6, index=freq, values=(Wc[freq] * 1.1).astype(Wc.dtype))
m = s.measure("H6 frequent rows x1.1")
h6_ok = decide(best, m, "H6 freq rows x1.1")
if h6_ok:
    best = m

# the remaining passes: refine whichever lever moved
left = s.budget_left
print("passes left", left)
if left >= 1:
    if h6_ok:
        h7 = s.hypothesis("a further x1.1 on the frequent rows (x1.21 total) continues the gain",
                          "CER falls a further 0.1-0.3 points or substitutions overtake: then revert",
                          "output rows (scale rows of frequent letters)")
        Wc = s.subject.tensors()[weight]
        s.edit("set_rows", [weight], hypothesis=h7, index=freq, values=(Wc[freq] * 1.1).astype(Wc.dtype))
        m = s.measure("H7 frequent rows x1.21")
        if decide(best, m, "H7 freq rows x1.21"):
            best = m
    else:
        h7 = s.hypothesis("the deletions are a frame-level blank preference, not row strength: blank bias -1.0 total is the stronger lever",
                          "CER falls 0.3-1.0 points or collapsed lines rise: then revert",
                          "output bias (blank row)")
        cur = float(s.subject.tensors()[bias][0])
        d = np.zeros_like(s.subject.tensors()[bias]); d[0] = (1.2323 - 1.0) - cur
        s.edit("add", [bias], hypothesis=h7, delta=d)
        m = s.measure("H7 blank bias -1.0 total")
        if decide(best, m, "H7 blank -1.0 total"):
            best = m

left = s.budget_left
if left >= 1:
    # a last confirmation pass is not needed (every kept state was measured); spend the rest on the combined
    # blank + space direction if neither alone was kept, else on a half step of the last kept lever
    kept = [e for e in s._log if e["kind"] == "decision" and e["kept"]]
    if not kept:
        h8 = s.hypothesis("blank -0.113 and space +0.5 together, each too small alone, add up",
                          "CER falls by 0.1-0.3 points; else the deletion reading is wrong for this model",
                          "output bias (blank and space rows)")
        d = np.zeros_like(s.subject.tensors()[bias]); d[0] = BLANK_DELTA_PARENT; d[space] = 0.5
        s.edit("add", [bias], hypothesis=h8, delta=d)
        m = s.measure("H8 blank -0.113 + space +0.5")
        if decide(best, m, "H8 blank+space"):
            best = m
    else:
        last = kept[-1]
        s.note(f"last kept edit: {last['edit']} ({last['reason']}); remaining {left} pass(es) left unspent on purpose")

summary = s.freeze()
print(json.dumps({k: summary[k] for k in ("used", "edits", "kept", "start_digest", "final_digest")}, indent=1))
print("baseline", summary["baseline"]["cer"], "last", summary["last_pass"]["cer"])
