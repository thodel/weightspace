"""planted-f1 / s12 / task (a), budget 10. Written in full before the first pass; see preregistration_hypotheses.md.

Primary hypothesis: the output rows of `e` and `n` were swapped (C0: 2,201 of 3,144 edits are e<->n; against the
parent only those two rows moved, cosine 0.36, with their norms exchanged). Secondary probes: C_3 bias, L_16 outliers,
space bias. Every edit is decided by s.keep (the harness rule) plus a collapsed-lines check.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"

s = open_session(".")
s.note("s12/a: programmer claude-fable-5-1; plan P1 baseline, P2 swap e/n rows, P3 bias swap or parent transplant, "
       "P4 C_3 bias, P5 L_16 outliers, P6 space bias (conditional)")

cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
pm = parent.codec_map()
PT = parent.tensors()


def T():
    return s.subject.tensors()


def conf(m, ref, hyp):
    """Count of one confusion in m.confusions; hyp None means 'nothing' (deletion)."""
    n = 0
    for c in m.confusions:
        r, h, k = c[0], c[1], c[2]
        if r == ref and (h == hyp or (hyp is None and (h in ("", "∅") or h is None))):
            n += int(k)
    return n


def decide(label, reason, before):
    """Measure, print, keep by the rule; refuse ourselves if the collapsed-line count rose."""
    m = s.measure(label)
    print(f"{label}: CER {m.cer:.4f} WER {m.wer:.4f} collapsed {m.collapsed} edits {m.char_edits}; top {m.confusions[:5]}")
    if before is not None and m.collapsed > before.collapsed + 2:
        s.revert(f"{reason}; collapsed lines rose {before.collapsed} -> {m.collapsed}")
        return False, m
    kept = s.keep(reason)
    print(f"  kept={kept}")
    return kept, m


# ---- P1: baseline ------------------------------------------------------------------------------------------------
m0 = s.measure("C0")
print(f"C0: CER {m0.cer:.4f} WER {m0.wer:.4f} collapsed {m0.collapsed}; top {m0.confusions[:5]}")
state = m0          # metrics of the last kept state
en_fixed = False

if "e" in cm and "n" in cm:
    ie, in_ = cm["e"], cm["n"]

    # ---- P2: H1a swap the weight rows of e and n back --------------------------------------------------------------
    h = s.hypothesis("H1a: output rows e and n of O_18.lin.weight are swapped (cos 0.36 to parent, norms exchanged; "
                     "e->n 1364, n->e 837 on C0)",
                     "CER falls by 0.15-0.19 (0.244 -> ~0.06-0.09); e<->n leaves the top confusions", "set_rows swap")
    w = T()[W]
    s.edit("set_rows", [W], hypothesis=h, index=[ie, in_], values=np.stack([w[in_].copy(), w[ie].copy()]))
    kept, m = decide("H1a swap e/n weight rows", "e/n rows swapped back", state)
    if kept:
        state = m
        en_fixed = True
        s.note(f"after H1a: e->n {conf(m, 'e', 'n')}, n->e {conf(m, 'n', 'e')} (C0: {conf(m0, 'e', 'n')}, {conf(m0, 'n', 'e')})")

        # ---- P3: H1b the biases were swapped too ------------------------------------------------------------------
        if s.budget_left > 0:
            h = s.hypothesis("H1b: the biases of e and n were swapped along with the rows",
                             "small, |dCER| < 0.005, direction unknown", "set_rows swap (bias)")
            b = T()[B]
            s.edit("set_rows", [B], hypothesis=h, index=[ie, in_], values=np.array([b[in_], b[ie]], dtype=b.dtype))
            kept, m = decide("H1b swap e/n biases", "e/n biases swapped back", state)
            if kept:
                state = m
    else:
        # ---- P3': H1c rows corrupted beyond a swap: transplant the parent's e and n rows ---------------------------
        if s.budget_left > 0 and "e" in pm and "n" in pm:
            h = s.hypothesis("H1c: e/n rows are corrupted beyond a swap; parent rows (aligned by character) as donor",
                             "CER falls by 0.08-0.15 (partial repair, parent rows ~0.2 from healthy child rows)",
                             "set_rows transplant (parent donor)")
            pw, pb = PT[W], PT[B]
            s.edit("set_rows", [W, B], hypothesis=h,
                   per_tensor={W: {"index": [ie, in_], "values": np.stack([pw[pm["e"]], pw[pm["n"]]])},
                               B: {"index": [ie, in_], "values": np.array([pb[pm["e"]], pb[pm["n"]]], dtype=pb.dtype)}})
            kept, m = decide("H1c parent e/n rows", "parent e/n rows transplanted", state)
            if kept:
                state = m
                en_fixed = True
            else:
                s.note("H1a and H1c both refused: the e/n defect is not in the values of those two output rows; O_18 left alone")
else:
    s.note("codec lacks 'e' or 'n' as single-label characters; H1 skipped")

# ---- P4: H2 C_3.co.bias from the parent ----------------------------------------------------------------------------
if s.budget_left > 0 and "C_3.co.bias" in PT and PT["C_3.co.bias"].shape == T()["C_3.co.bias"].shape:
    h = s.hypothesis("H2: C_3.co.bias perturbed (rel L2 1.09, cos 0.68 to parent, other conv biases 0.15-0.33)",
                     "|dCER| < 0.003, probably not kept", "transplant (parent donor)")
    s.edit("transplant", ["C_3.co.bias"], hypothesis=h, donor=PT["C_3.co.bias"])
    kept, m = decide("H2 C_3 bias from parent", "C_3 bias transplanted", state)
    if kept:
        state = m

# ---- P5: H3 L_16 weight_ih_l0 outliers back to the parent's values --------------------------------------------------
name = "L_16.layer.weight_ih_l0"
if s.budget_left > 0 and name in PT and PT[name].shape == T()[name].shape:
    w = T()[name]
    mask = np.abs(w) > 3.0
    n_out = int(mask.sum())
    print(f"L_16 ih outliers |w|>3.0: {n_out}")
    if n_out > 0:
        h = s.hypothesis(f"H3: {n_out} entries of L_16.layer.weight_ih_l0 with |w|>3.0 (abs max 4.70 vs ~2 elsewhere) are blown up",
                         "|dCER| < 0.003, probably not kept", "add (outliers to parent values)")
        delta = np.where(mask, PT[name] - w, 0.0).astype(w.dtype)
        s.edit("add", [name], hypothesis=h, delta=delta)
        kept, m = decide("H3 L_16 outliers", "L_16 ih outliers returned to parent", state)
        if kept:
            state = m
    else:
        s.note("H3: no entry above 3.0 in L_16.layer.weight_ih_l0; skipped")

# ---- P6: H4 space bias, only if space deletions are still frequent -------------------------------------------------
if s.budget_left > 0 and " " in cm and conf(state, " ", None) >= 50:
    isp = cm[" "]
    h = s.hypothesis(f"H4: space bias {float(T()[B][isp]):.2f} (most negative of all rows) causes the {conf(state, ' ', None)} space deletions",
                     "CER falls 0.002-0.006 if true; otherwise it rises slightly and the rule refuses", "add (bias, +1.5)")
    delta = np.zeros_like(T()[B])
    delta[isp] = 1.5
    s.edit("add", [B], hypothesis=h, delta=delta)
    kept, m = decide("H4 space bias +1.5", "space bias raised", state)
    if kept:
        state = m

s.note(f"end: e/n repaired={en_fixed}; last kept CER {state.cer:.4f} (C0 {m0.cer:.4f}); passes left {s.budget_left}")
summary = s.freeze()
print({k: summary[k] for k in ("used", "edits", "kept", "final_digest")})
