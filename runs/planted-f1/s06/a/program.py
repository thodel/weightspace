"""planted-f1 / s06 / task (a), budget 10 — the programmer's script (claude-fable-5-1).

Written in full before the first pass; see preregistration_hypotheses.md (H1–H7) in this folder.
Runs inside the sandbox: wsp + numpy only. Every edit is decided by s.keep (the harness's rule);
the program branches only on m.cer / m.collapsed / m.confusions / m.length_ratio.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"
NOTHING = {"", "∅", None, "<nothing>", "ε", "<eps>", "<del>", "<ins>"}

s = open_session(".")
cm = s.subject.codec_map()


# ---------------------------------------------------------------- helpers ----------------------------------------
def tensors():
    return s.subject.tensors()           # re-read after every edit: set_tensor replaces the array


def _clean(x):
    if x in NOTHING:
        return ""
    if isinstance(x, str) and len(x) > 1 and x.strip(" ") != "":
        return x.strip(" ")
    return x


def conf_pairs(m):
    """-> list of (ref, hyp, count); robust to the container Metrics.confusions uses. '' = nothing."""
    c = getattr(m, "confusions", None)
    out = []
    try:
        if isinstance(c, dict):
            for k, v in c.items():
                if isinstance(k, tuple) and len(k) == 2:
                    out.append((_clean(k[0]), _clean(k[1]), int(v)))
                elif isinstance(k, str):
                    for sep in ("→", "->", "\t", "|", "=>"):
                        if sep in k:
                            a, b = k.split(sep, 1)
                            out.append((_clean(a), _clean(b), int(v)))
                            break
        elif isinstance(c, (list, tuple)):
            for item in c:
                if isinstance(item, dict):
                    r = item.get("ref", item.get("reference", item.get("a")))
                    h = item.get("hyp", item.get("hypothesis", item.get("b")))
                    n = item.get("count", item.get("n", item.get("k", 0)))
                    out.append((_clean(r), _clean(h), int(n)))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    if len(item) == 2 and isinstance(item[0], (list, tuple)) and len(item[0]) == 2:
                        out.append((_clean(item[0][0]), _clean(item[0][1]), int(item[1])))
                    elif len(item) >= 3:
                        out.append((_clean(item[0]), _clean(item[1]), int(item[2])))
    except Exception as ex:              # never let the bookkeeping kill the session
        s.note(f"conf_pairs: could not parse confusions ({type(c).__name__}): {ex}")
    return out


def count(m, ref, hyp):
    return sum(n for r, h, n in conf_pairs(m) if r == ref and h == hyp)


def count_from(m, ref):
    return sum(n for r, h, n in conf_pairs(m) if r == ref)


def top(m, k=8):
    return sorted(conf_pairs(m), key=lambda t: -t[2])[:k]


def lr(m):
    v = getattr(m, "length_ratio", None)
    return float(v) if v is not None else float("nan")


def report(label, m):
    s.note(f"{label}: CER {m.cer:.4f} WER {m.wer:.4f} collapsed {m.collapsed} length_ratio {lr(m):.3f} "
           f"e->nothing {count(m, 'e', '')} nothing->e {count(m, '', 'e')} e-errors {count_from(m, 'e')} "
           f"space->nothing {count(m, ' ', '')} nothing->space {count(m, '', ' ')} top {top(m)}")
    print(label, round(m.cer, 4), m.collapsed)


def trial(text, prediction, edit_class, op, names, label, reason, **kwargs):
    """hypothesis -> edit -> one pass -> s.keep (the rule decides). Returns (metrics or None, kept)."""
    if s.budget_left < 1:
        s.note(f"no pass left; skipping {label}")
        return None, False
    h = s.hypothesis(text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, **kwargs)
    m = s.measure(label)
    kept = s.keep(reason)
    report(f"{label} (kept={kept})", m)
    return m, kept


def bias_delta(row, value):
    d = np.zeros(tensors()[B].shape, dtype=np.float32)
    d[row] = value
    return d


# ---------------------------------------------------------------- donor -------------------------------------------
parent = None
try:
    parent = KrakenSubject(PARENT)
    pm = parent.codec_map()
    PT = parent.tensors()
    s.note(f"donor loaded: parent codec {len(pm)} chars, 'e' row {pm.get('e')}, "
           f"parent e-row norm {np.linalg.norm(PT[W][pm['e']]):.3f}, bias {PT[B][pm['e']]:.3f}")
except Exception as ex:
    s.note(f"parent could not be loaded as donor: {ex}; donor edits (H1, H2, H4) are skipped")

e_row = cm.get("e")
e_norm0 = float(np.linalg.norm(tensors()[W][e_row])) if e_row is not None else float("nan")
s.note(f"subject: 'e' row {e_row}, norm {e_norm0:.4f}, bias {tensors()[B][e_row]:.3f}; "
       f"C_3.co.bias norm {np.linalg.norm(tensors()['C_3.co.bias']):.4f}; budget {s.budget_left}")


def mapped_e_row(n_rows=80):
    """H2: the parent's e row mapped into the child's output space.
    Fit A (400x400, ridge towards 0) on the n_rows weakest shared rows so that P (I + A) ~ C;
    return pe (I + A). Fallback: pe + mean(C - P)."""
    Wc, Wp = tensors()[W].astype(np.float64), PT[W].astype(np.float64)
    shared = [c for c in cm if c in pm and c != "e"]
    norms = np.array([np.linalg.norm(Wp[pm[c]]) for c in shared])
    sel = [shared[i] for i in np.argsort(norms)[:n_rows]]
    P = np.stack([Wp[pm[c]] for c in sel])
    C = np.stack([Wc[cm[c]] for c in sel])
    pe = Wp[pm["e"]]
    G = P.T @ P
    lam = float(np.trace(G)) / G.shape[0]
    A = np.linalg.solve(G + lam * np.eye(G.shape[0]), P.T @ (C - P))
    e_hat = pe + pe @ A
    ratio = np.linalg.norm(e_hat) / max(np.linalg.norm(pe), 1e-9)
    how = "ridge"
    if not (0.6 <= ratio <= 1.8) or not np.all(np.isfinite(e_hat)):
        e_hat = pe + (C - P).mean(axis=0)
        how = "mean-shift fallback"
    fit_before = float(np.linalg.norm(C - P) / np.linalg.norm(C))
    fit_after = float(np.linalg.norm(C - (P + P @ A)) / np.linalg.norm(C))
    s.note(f"H2 map ({how}): {len(sel)} rows, lambda {lam:.2f}, rel residual {fit_before:.3f} -> {fit_after:.3f}, "
           f"|e_hat|/|pe| {ratio:.3f}, cos(e_hat, pe) {float(e_hat @ pe / (np.linalg.norm(e_hat) * np.linalg.norm(pe))):.3f}")
    return e_hat.astype(np.float32)


# ---------------------------------------------------------------- passes ------------------------------------------
# pass 1: C0 baseline
m0 = s.measure("C0")
report("C0", m0)
cur = m0                                  # metrics of the current kept state

# H1: parent e row transplanted (the row was zeroed)
if parent is not None and e_row is not None and "e" in pm and e_norm0 < 0.5:
    m, kept = trial(
        "H1: output row 'e' of O_18.lin.weight is all zero (norm 0.00; parent 7.64, cosine 0): the row was zeroed. "
        "The parent's 'e' row, aligned by character, restores the class.",
        "CER falls by 0.07-0.11 (0.162 -> 0.05-0.09); e->nothing from 910 to <150; length ratio >= 0.96; fewer collapsed lines",
        "donor row transplant (set_rows, O_18.lin.weight row e <- parent row e)",
        "set_rows", [W], "H1_e_row_parent", "zeroed e row restored from the parent",
        index=[e_row], values=PT[W][pm["e"]].astype(np.float32))
    if kept:
        cur = m
    elif m is not None:
        s.note(f"H1 refused: delta CER {m.cer - cur.cer:+.4f}; the raw parent row does not fit the child's features (or the defect is elsewhere)")
else:
    s.note("H1 skipped: no donor, or the e row is not zero (hypothesis void)")

# H2: parent e row mapped into the child's output space
if parent is not None and e_row is not None and "e" in pm and s.budget_left >= 1:
    try:
        e_hat = mapped_e_row()
    except Exception as ex:
        e_hat = None
        s.note(f"H2 skipped: mapping failed ({ex})")
    if e_hat is not None:
        m, kept = trial(
            "H2: the child's other output rows moved ~0.2 rel from the parent's (L_16 drifted 0.3); the donor 'e' row "
            "fits better after a ridge-regularised linear map fitted on the 80 weakest shared rows (parent row -> child row).",
            "a further fall of 0.005-0.02 CER over H1 (0.04-0.07); if refused the raw parent row is already as good as measurable",
            "donor row, adapted (set_rows with parent_e @ (I + A))",
            "set_rows", [W], "H2_e_row_mapped", "mapped donor row beats the raw transplant",
            index=[e_row], values=e_hat)
        if kept:
            cur = m

# H3: e bias nudge, direction from the residual (up to two steps of +-0.5)
if e_row is not None and s.budget_left >= 1 and count_from(cur, "e") > 0:
    for step in (1, 2):
        d_e, i_e = count(cur, "e", ""), count(cur, "", "e")
        if d_e > 2 * i_e + 10:
            delta = +0.5
        elif i_e > 2 * d_e + 10:
            delta = -0.5
        else:
            s.note(f"H3 step {step} skipped: e deletions {d_e} vs insertions {i_e} are balanced")
            break
        m, kept = trial(
            f"H3 step {step}: after the row repair 'e' still loses to the blank (e->nothing {d_e}, nothing->e {i_e}); "
            f"the class threshold is off by a constant, so the bias moves {delta:+.1f}.",
            "CER down 0.002-0.01 and the e deletion/insertion imbalance shrinks; if refused the remaining e errors are directional, not a threshold",
            "bias nudge (add on O_18.lin.bias[e])",
            "add", [B], f"H3_e_bias_{step}_{delta:+.1f}", "e bias nudge improves e/blank balance",
            delta=bias_delta(e_row, delta))
        if kept:
            cur = m
        else:
            break
        if s.budget_left < 1:
            break

# H4: C_3.co.bias from the parent (its relative move 1.09 / cosine 0.68 is out of line with every other block)
if parent is not None and s.budget_left >= 1 and tensors()["C_3.co.bias"].shape == PT["C_3.co.bias"].shape:
    m, kept = trial(
        "H4: C_3.co.bias moved rel 1.09 (cosine 0.68) from the parent while C_3.co.weight moved 0.17 and every other "
        "block's bias moved as much as its weight: noise of about the parent's own norm was added to the 32-vector. "
        "The parent's bias is the better estimate.",
        "CER down 0.002-0.015 if the bias was noised; otherwise no change or a tiny rise (refused): then it is fine-tuning drift of a near-zero vector",
        "donor transplant (C_3.co.bias <- parent)",
        "transplant", ["C_3.co.bias"], "H4_C3_bias_parent", "parent C_3 bias replaces a noised one",
        donor=PT["C_3.co.bias"].astype(np.float32))
    if kept:
        cur = m

# H5: space bias, only if spaces are still clearly under-predicted
sp = cm.get(" ")
if sp is not None and s.budget_left >= 1:
    d_sp, i_sp = count(cur, " ", ""), count(cur, "", " ")
    if d_sp > i_sp + 20:
        m, kept = trial(
            f"H5: space has the most negative bias (-7.85) and is deleted {d_sp} times vs {i_sp} insertions: under-predicted by a constant.",
            "CER down <= 0.005, WER down more; if refused the deletions are the model's ordinary error, not a bias defect",
            "bias nudge (add +0.75 on O_18.lin.bias[' '])",
            "add", [B], "H5_space_bias_+0.75", "space bias nudge reduces space deletions",
            delta=bias_delta(sp, 0.75))
        if kept:
            cur = m
    else:
        s.note(f"H5 skipped: space deletions {d_sp} vs insertions {i_sp}")

# H6: blank bias, only if lines still collapse
if s.budget_left >= 1:
    if cur.collapsed >= 4:
        m, kept = trial(
            f"H6: {cur.collapsed} lines still collapse after the e repair: the blank (bias +1.23, the only positive large bias) dominates on them.",
            "collapsed -> <= 2 and CER down <= 0.005; if refused the collapsed lines are not a blank problem (unreadable slivers or hard lines)",
            "bias nudge (add -0.5 on O_18.lin.bias[0])",
            "add", [B], "H6_blank_bias_-0.5", "blank bias nudge un-collapses lines",
            delta=bias_delta(0, -0.5))
        if kept:
            cur = m
    else:
        s.note(f"H6 skipped: collapsed {cur.collapsed} < 4")

# H7 (leftover passes): the e row is right in direction but too short
if e_row is not None and s.budget_left >= 2 and count(cur, "e", "") > 100:
    row_now = tensors()[W][e_row].astype(np.float32)
    if np.linalg.norm(row_now) > 0:
        m, kept = trial(
            f"H7: e->nothing is still {count(cur, 'e', '')} with the bias settled: the recovered row points right but is too short for the child's feature scale.",
            "CER down <= 0.005; if refused the row length is fine and the remaining e errors are directional",
            "row scale (set_rows x1.2 of the current e row)",
            "set_rows", [W], "H7_e_row_x1.2", "longer e row wins more frames from the blank",
            index=[e_row], values=(row_now * 1.2).astype(np.float32))
        if kept:
            cur = m
else:
    s.note(f"H7 skipped: budget_left {s.budget_left}, e->nothing {count(cur, 'e', '')}")

s.note(f"end: CER {cur.cer:.4f} (baseline {m0.cer:.4f}), collapsed {cur.collapsed}, passes left {s.budget_left}")
summary = s.freeze()
print({k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})
