"""The programmer's script for session 20261106-bern-combined-v1-b10-1 (target 1, re-framed).

Question: kraken-bern-combined-v1 was fine-tuned from hertwig-v2 on 168 pages of which 44
carried the wrong image (docs/HERTWIG_REVISIONS.md). Is the damage localised, and can a
weight-space edit take it out without losing what the clean Schlussredaktion pages taught?

Donors, both read-only:
  V2 = kraken-bern-hertwig-v2         the parent, before the noisy fine-tune
  SR = kraken-bern-schlussredaktion-v1  the clean sibling: same parent, fine-tuned on the clean
                                        Schlussredaktion pages only (24.10 export)
Output rows are aligned by character (combined has ½ ƒ ̊; SR has ½ ̊; V2 has none of them);
rows a donor lacks keep combined's values.

Keep rule: Session.keep() (PROTOCOL §3, 06.11.2026): CER down against the last kept state with the
paired 95 % interval below 0; the programmer additionally reverts any edit that raises the
collapsed count. One hypothesis per pass, written before the edit. Budget 10; unspent passes stay
unspent. The whole plan was written before the sandbox run (preregistration.md).
"""
import json
import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

ROOT = "/home/tobias/Repo/weightspace/models/kraken/"
s = open_session(".")
V2 = KrakenSubject(ROOT + "kraken-bern-hertwig-v2.mlmodel")
SR = KrakenSubject(ROOT + "kraken-bern-schlussredaktion-v1.mlmodel")
T = lambda: s.subject.tensors()
cc = s.subject.codec_map()
W, B = "O_18.lin.weight", "O_18.lin.bias"
print(s.subject.describe()["parameters"], "parameters;", s.budget_left, "passes left")

state = {"kept": None}


def decide(m, what):
    """Collapsed lines must not rise (programmer's check), then the harness's rule."""
    ref = state["kept"]
    if m.collapsed > ref.collapsed:
        s.revert(f"{what}: collapsed {ref.collapsed} -> {m.collapsed}")
        print("REVERT (collapsed)", what, m.cer, m.collapsed)
        return False
    ok = s.keep(f"{what}: CER {ref.cer:.4f} -> {m.cer:.4f}, collapsed {ref.collapsed} -> {m.collapsed}")
    print("KEEP" if ok else "REFUSED", what, f"{ref.cer:.5f} -> {m.cer:.5f}", m.collapsed, s._vs_kept["ci95"])
    if ok:
        state["kept"] = m
    return ok


def block(prefix):
    return [n for n in T() if n.startswith(prefix + ".")]


def transplant_block(names, donor, hyp):
    s.edit("transplant", names, hypothesis=hyp,
           per_tensor={n: {"donor": donor.tensors()[n]} for n in names})


def aligned_output(donor):
    dm = donor.codec_map()
    w, b = T()[W].copy(), T()[B].copy()
    dw, db = donor.tensors()[W], donor.tensors()[B]
    w[0], b[0] = dw[0], db[0]                       # the blank
    shared = 0
    for ch, row in cc.items():
        if ch in dm:
            w[row], b[row] = dw[dm[ch]], db[dm[ch]]
            shared += 1
    return w, b, shared


# pass 1: C0
state["kept"] = s.measure("C0 baseline")
print("C0", state["kept"].cer, state["kept"].collapsed, state["kept"].confusions[:5])
kept = {}

# H1: the readings' first hypothesis for this pair
h = s.hypothesis("the noisy fine-tune raised the blank bias by +0.118 (0.879 -> 0.998); putting it back to hertwig-v2's value reduces deletions",
                 "practice CER -0.2 to -1.0 points (docs/READINGS.md said -1 to -2 on Hertwig-only lines; phase 1 found blank edits noise-sized on xix-blb); collapsed not up",
                 "output bias (blank row)")
d = np.zeros_like(T()[B]); d[0] = float(V2.tensors()[B][0] - T()[B][0])
s.edit("add", [B], hypothesis=h, delta=d)
kept["H1"] = decide(s.measure("H1 blank bias to hertwig-v2"), "H1 blank to v2")

# H2-H6: the clean sibling, one block at a time, output layer first, then down the network
plan = [("H2", "O_18", "output layer (weight and bias, rows aligned by character; ƒ kept)"),
        ("H3", "L_16", "last BiLSTM"), ("H4", "L_14", "middle BiLSTM"), ("H5", "L_12", "first BiLSTM"),
        ("H6", "C_", "conv stack C_0-C_9")]
for tag, prefix, what in plan:
    h = s.hypothesis(f"the label noise damaged the {what}; the clean sibling's {what} (same parent, clean Schlussredaktion pages) reads both hands better",
                     "practice CER falls with an interval below 0 if the damage sits here; Schlussredaktion lines must not get worse on balance (they are 2/3 of the practice set)",
                     f"transplant from schlussredaktion-v1: {prefix}")
    if prefix == "O_18":
        w, b, shared = aligned_output(SR)
        s.edit("transplant", [W, B], hypothesis=h, per_tensor={W: {"donor": w}, B: {"donor": b}})
        s.note(f"H2: {shared} of {len(cc)} character rows + blank from schlussredaktion-v1")
    else:
        names = [n for n in T() if n.startswith(prefix)] if prefix == "C_" else block(prefix)
        transplant_block(names, SR, h)
    kept[tag] = decide(s.measure(f"{tag} {prefix} from schlussredaktion-v1"), f"{tag} {prefix} <- SR")

# H7: the readings' hypothesis (b) with the parent as donor, if the sibling's output layer was not kept
if not kept["H2"] and s.budget_left:
    h = s.hypothesis("the output layer from hertwig-v2 (rows aligned, the 3 new rows kept) restores the Hertwig reading while the recurrent stack carries Schlussredaktion",
                     "Hertwig lines gain, Schlussredaktion lines lose; net practice CER -0.5 to -2 points if the damage is in the classifier",
                     "transplant from hertwig-v2: O_18")
    w, b, shared = aligned_output(V2)
    s.edit("transplant", [W, B], hypothesis=h, per_tensor={W: {"donor": w}, B: {"donor": b}})
    kept["H7"] = decide(s.measure("H7 O_18 from hertwig-v2"), "H7 O_18 <- V2")

# H8: the gap is on the Hertwig hand only (bern-eval: combined 26.7 %, hertwig-v2 23.6 %; Schlussredaktion
# 13.5 % vs 30.6 %). The xix-blb alpha-sweep found the last BiLSTM carries a hand's spacing. Half of L_16's
# fine-tune move undone toward the parent may give back Hertwig spacing at a smaller Schlussredaktion cost.
if s.budget_left:
    names = block("L_16")
    h = s.hypothesis("the Hertwig loss sits in the last BiLSTM: L_16 half-way back toward hertwig-v2 recovers Hertwig lines at a smaller Schlussredaktion cost",
                     "practice CER inside [-0.5, +1.0] points: Hertwig lines -1 to -3 points, Schlussredaktion lines +0.5 to +2; most likely refused",
                     "interpolate alpha 0.5 toward hertwig-v2 on L_16")
    s.edit("interpolate", names, hypothesis=h, alpha=0.5, per_tensor={n: {"other": V2.tensors()[n]} for n in names})
    kept["H8"] = decide(s.measure("H8 L_16 alpha 0.5 to V2"), "H8 L_16 half-way to V2")

s.note(f"decisions: {kept}; {s.budget_left} pass(es) left unspent on purpose")
summary = s.freeze()
print(json.dumps({k: summary[k] for k in ("used", "edits", "kept", "start_digest", "final_digest")}, indent=1))
print("baseline", summary["baseline"]["cer"], "last", summary["last_pass"]["cer"], "changed", list(summary["changed_tensors"]))
