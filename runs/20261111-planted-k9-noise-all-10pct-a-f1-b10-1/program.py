"""planted-f1 / s10 / task (a): programmer with 10 passes.

Hypotheses and predictions: preregistration_hypotheses.md (written before this file, not changed after).
Order: C0 baseline; H1 hidden-tensor structural repair (conditional); H2 output-weight repair (conditional);
H3 output-bias repair, space first (conditional); H4 blank-bias sweep; H5 blank row from donor; H6 C_3 bias control.
Every edit: hypothesis -> edit -> measure -> keep (the harness's rule decides). At most 10 measures.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
MAX_MEASURES = 10

s = open_session(".")
parent = KrakenSubject(PARENT)
P = {k: np.asarray(v, dtype=np.float64) for k, v in parent.tensors().items()}
pm = parent.codec_map()
cm = s.subject.codec_map()

used = 0
pending = False      # an edit awaiting keep/revert


def T(name):
    """Current value of a subject tensor (re-read after every edit)."""
    return np.asarray(s.subject.tensors()[name], dtype=np.float64)


def left():
    return min(s.budget_left, MAX_MEASURES - used)


# ----------------------------------------------------------------------------- metrics helpers
def del_ins(m):
    """(deletions, insertions) from m.confusions, tolerant of its container shape; (None, None) if unreadable."""
    conf = getattr(m, "confusions", None)
    if conf is None:
        return None, None
    nothing = {"", None, "∅", "<eps>", "<nothing>"}
    dels = ins = 0
    try:
        if isinstance(conf, dict):
            items = []
            for k, v in conf.items():
                if isinstance(k, (tuple, list)) and len(k) >= 2:
                    items.append((k[0], k[1], v))
        else:
            items = []
            for x in conf:
                if isinstance(x, dict):
                    ref = x.get("ref", x.get("reference"))
                    hyp = x.get("hyp", x.get("hypothesis"))
                    n = x.get("count", x.get("n", 0))
                    items.append((ref, hyp, n))
                else:
                    t = tuple(x)
                    if len(t) >= 3:
                        items.append((t[0], t[1], t[2]))
        for ref, hyp, n in items:
            n = int(n)
            if hyp in nothing and ref not in nothing:
                dels += n
            elif ref in nothing and hyp not in nothing:
                ins += n
        return dels, ins
    except Exception as e:      # unknown shape: fall back to collapsed/CER checks only
        s.note(f"confusions not parsed: {e!r}")
        return None, None


def still_deleting(m):
    """Deletion-dominated reading: lines collapse or deletions outnumber insertions 2:1."""
    if m.collapsed > 5:
        return True
    d, i = del_ins(m)
    if d is None:
        lr = getattr(m, "length_ratio", None)
        return lr is not None and lr < 0.9
    return d > 2 * max(i, 1)


# ----------------------------------------------------------------------------- the trial
def trial(label, text, prediction, edit_class, op, names, reason, **kw):
    """hypothesis -> edit -> one pass -> keep (rule) ; returns (kept, metrics) or (None, None) when no pass is left."""
    global used, pending
    if left() < 1:
        s.note(f"skipped, no pass left: {label}")
        return None, None
    h = s.hypothesis(text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, **kw)
    pending = True
    try:
        m = s.measure(label)
    except Exception as e:
        s.revert(f"measure failed: {e!r}")
        pending = False
        raise
    used += 1
    kept = s.keep(reason)
    pending = False
    d, i = del_ins(m)
    s.note(f"{label}: CER {m.cer:.4f} WER {m.wer:.4f} collapsed {m.collapsed} del/ins {d}/{i} kept={kept}")
    return kept, m


# ----------------------------------------------------------------------------- scans against the parent
def robust(x):
    x = np.asarray(x, np.float64).ravel()
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, 1.4826 * mad + 1e-9


def sparse_outliers(c, p, k=12.0, floor=0.5):
    """Entries whose fine-tuning delta is far outside the tensor's own delta distribution."""
    d = c - p
    med, sig = robust(d)
    return np.abs(d - med) > max(k * sig, floor)


def scan_rows(c2, p2, lo, hi, zero=0.05):
    """Rows of c2 whose norm ratio to p2's row is < lo x or > hi x the median ratio: zeroed rows get the parent row
    (times the median ratio), scaled rows are rescaled to median ratio x parent norm. Returns (repaired, n_flagged)."""
    out = c2.copy()
    nc = np.linalg.norm(out, axis=1)
    npn = np.linalg.norm(p2, axis=1)
    valid = npn > 0.2 * np.median(npn)
    if int(valid.sum()) < 4:
        return out, 0
    ratio = nc / (npn + 1e-12)
    rho = float(np.median(ratio[valid]))
    bad = valid & ((ratio < lo * rho) | (ratio > hi * rho))
    for i in np.where(bad)[0]:
        if nc[i] < zero * npn[i]:
            out[i] = p2[i] * rho
        else:
            out[i] *= rho * npn[i] / nc[i]
    return out, int(bad.sum())


def structural_repair(c, p, lo, hi):
    """Rows then columns (2-D), or out-channels then in-channels (conv 4-D). Returns (repaired, n_rows, n_cols)."""
    if c.ndim == 2:
        r, nr = scan_rows(c, p, lo, hi)
        rt, ncol = scan_rows(r.T, p.T, lo, hi)
        return rt.T.copy(), nr, ncol
    if c.ndim == 4:
        o, i_, kh, kw = c.shape
        r, nr = scan_rows(c.reshape(o, -1), p.reshape(o, -1), lo, hi)
        r = r.reshape(c.shape)
        rt, ncol = scan_rows(r.transpose(1, 0, 2, 3).reshape(i_, -1), p.transpose(1, 0, 2, 3).reshape(i_, -1), lo, hi)
        r = rt.reshape(i_, o, kh, kw).transpose(1, 0, 2, 3)
        return np.ascontiguousarray(r), nr, ncol
    return c.copy(), 0, 0


def main():
    global pending
    # ---------------------------------------------------------------- pass 1: baseline
    global used
    m0 = s.measure("C0")
    used += 1
    d0, i0 = del_ins(m0)
    s.note(f"C0: CER {m0.cer:.4f} WER {m0.wer:.4f} collapsed {m0.collapsed} del/ins {d0}/{i0}")
    last = m0

    # ---------------------------------------------------------------- H1: hidden tensors vs parent
    hidden = [n for n in s.subject.tensors() if not n.startswith("O_18") and "bias_ih" not in n]
    deltas, report = {}, []
    for n in hidden:
        c = T(n)
        p = P.get(n)
        if p is None or p.shape != c.shape:
            continue
        rep, nr, ncol = structural_repair(c, p, 0.5, 2.0) if c.ndim >= 2 else (c.copy(), 0, 0)
        mask = sparse_outliers(rep, p)
        nout = int(mask.sum())
        if nout >= 3:
            rep[mask] = p[mask]
        else:
            nout = 0
        if nr or ncol or nout:
            deltas[n] = (rep - c).astype(np.float32)
            report.append((n, nr, ncol, nout, float(np.linalg.norm(rep - c))))
    s.note("H1 scan: " + ("; ".join(f"{n}: rows {nr} cols {nc} entries {no} |delta| {d:.2f}" for n, nr, nc, no, d in report) or "nothing flagged"))
    if deltas:
        kept, m = trial("H1-hidden-structural",
                        "hidden tensors carry planted damage (sparse large entries, zeroed or rescaled rows/columns) visible as "
                        "outliers against the parent: " + ", ".join(n for n in deltas),
                        "CER 0.55 -> below 0.25 (delta <= -0.30) if this is the main defect; collapsed lines 123 -> < 30",
                        "set entries / rows / columns to donor (parent) values or norms",
                        "add", list(deltas), "restores the parent's structure where the child is anomalous",
                        per_tensor={n: {"delta": d} for n, d in deltas.items()})
        if kept:
            last = m
        elif kept is False and len(deltas) > 1:
            worst = max(report, key=lambda r: r[4])[0]
            kept, m = trial("H1b-single-tensor",
                            f"the combined repair was refused; the single most-anomalous tensor {worst} alone carries the defect",
                            "delta <= -0.10 if that tensor is the defect; else refused",
                            "set entries / rows / columns to donor values or norms (one tensor)",
                            "add", [worst], "single-tensor structural repair", delta=deltas[worst])
            if kept:
                last = m
    else:
        s.hypothesis("H1 (not triggered): no hidden tensor has sparse/row/column anomalies against the parent",
                     "no edit, no pass", "none")

    # ---------------------------------------------------------------- output layer alignment
    W = T("O_18.lin.weight")
    b = T("O_18.lin.bias")
    Wp, bp = P["O_18.lin.weight"], P["O_18.lin.bias"]
    shared = [ch for ch in cm if ch in pm and cm[ch] < W.shape[0] and pm[ch] < Wp.shape[0]]
    ic = np.array([0] + [cm[ch] for ch in shared])
    ip = np.array([0] + [pm[ch] for ch in shared])
    chars = ["<blank>"] + shared
    s.note(f"output rows aligned by character: {len(shared)} shared + blank")

    # ---------------------------------------------------------------- H2: output weights (rows / columns / entries)
    Wc, Wps = W[ic], Wp[ip]
    rep, nr, ncol = structural_repair(Wc, Wps, 0.62, 1.6)
    mask = sparse_outliers(rep, Wps)
    nout = int(mask.sum())
    if nout >= 3:
        rep[mask] = Wps[mask]
    else:
        nout = 0
    ratio = np.linalg.norm(Wc, axis=1) / (np.linalg.norm(Wps, axis=1) + 1e-12)
    rho = float(np.median(ratio))
    flagged_rows = [chars[i] for i in range(len(chars)) if ratio[i] < 0.62 * rho or ratio[i] > 1.6 * rho]
    s.note(f"H2 scan: median norm ratio {rho:.3f}; rows flagged {flagged_rows}; cols {ncol}; entries {nout}")
    if nr or ncol or nout:
        dW = np.zeros_like(W)
        dW[ic] = rep - Wc
        kept, m = trial("H2-output-weights",
                        f"output rows/columns of O_18.lin.weight were altered (rows {flagged_rows}, {ncol} columns, {nout} entries)",
                        "delta -0.05 to -0.30; deletions of the affected characters drop by half",
                        "rescale rows/columns to donor norm, reset entries to donor",
                        "add", ["O_18.lin.weight"], "output rows/columns back to the parent's norm pattern",
                        delta=dW.astype(np.float32))
        if kept:
            last = m
    else:
        s.hypothesis("H2 (not triggered): output weight rows/columns match the parent's norm pattern", "no edit, no pass", "none")

    # ---------------------------------------------------------------- H3: output biases (space first)
    b = T("O_18.lin.bias")
    db = b[ic] - bp[ip]
    F = [ch for ch in "enisratdhulcgmobwfkz" if ch in shared]
    idxF = [chars.index(ch) for ch in F]
    muF, sigF = robust(db[idxF]) if len(idxF) >= 5 else (float(np.median(db[1:])), 1.0)
    punct = set(" .,-:;!?'\"()")
    flagged = {}
    for i, ch in enumerate(chars):
        if i == 0:
            continue
        dev = db[i] - muF
        if ch in F or ch in punct:
            if abs(dev) > max(4 * sigF, 2.0):
                flagged[ch] = (float(b[ic[i]]), float(bp[ip[i]] + muF))
        elif ch.isupper():
            if abs(dev) > max(4 * sigF, 3.5):
                flagged[ch] = (float(b[ic[i]]), float(bp[ip[i]] + muF))
        elif dev > 4.0:
            flagged[ch] = (float(b[ic[i]]), float(bp[ip[i]] + muF))
    s.note(f"H3 scan: mu_F {muF:+.3f} sigma_F {sigF:.3f}; space delta {db[chars.index(' ')] if ' ' in chars else float('nan'):+.3f}; "
           f"flagged {{{', '.join(f'{repr(k)}: {v[0]:+.2f}->{v[1]:+.2f}' for k, v in flagged.items())}}}")
    if flagged:
        dbias = np.zeros_like(b)
        for ch, (old, new) in flagged.items():
            dbias[cm[ch]] = new - old
        kept, m = trial("H3-output-bias",
                        f"output biases planted away from the fine-tuning pattern: {sorted(flagged)} (space -7.85 is the most negative of 297)",
                        "delta -0.03 to -0.08 if space is among them (1001 space deletions = 7.8 % of chars); WER down more",
                        "bias set to parent bias + median frequent-letter delta",
                        "add", ["O_18.lin.bias"], "biases back to the parent + fine-tuning offset",
                        delta=dbias.astype(np.float32))
        if kept:
            last = m
    else:
        s.hypothesis("H3 (not triggered): no output bias deviates from the fine-tuning pattern", "no edit, no pass", "none")

    # ---------------------------------------------------------------- H4: blank-bias sweep (symptom repair)
    steps = [-1.0, -1.5, -2.0]
    cum = 0.0
    for k, step in enumerate(steps):
        if not still_deleting(last):
            s.note(f"H4: reading no longer deletion-dominated after {cum:+.1f}; sweep stops")
            break
        dbl = np.zeros_like(b)
        dbl[0] = step
        kept, m = trial(f"H4-blank{cum + step:+.1f}",
                        f"the blank out-votes the characters (deletions only, {last.collapsed} collapsed lines): blank bias {step:+.1f} (cumulative {cum + step:+.1f})",
                        "delta -0.05 to -0.20 on the first step, collapsed 123 -> < 60; smaller on later steps",
                        "bias add on row 0",
                        "add", ["O_18.lin.bias"], "fewer blank frames, fewer deletions", delta=dbl.astype(np.float32))
        if kept is None:
            break
        if kept:
            cum += step
            last = m
            continue
        if k == 0 and last.collapsed > 40:
            dbl = np.zeros_like(b)
            dbl[0] = -3.0
            kept, m = trial("H4-blank-3.0",
                            "a -1.0 blank shift was refused but lines still collapse: the margin is large, test one big step of -3.0",
                            "delta <= -0.10 if the collapse is marginal at a larger scale; else refused (features dead)",
                            "bias add on row 0",
                            "add", ["O_18.lin.bias"], "big-step blank test", delta=dbl.astype(np.float32))
            if kept:
                cum += -3.0
                last = m
        break

    # ---------------------------------------------------------------- H5: blank row weights from the donor
    if left() >= 1:
        W = T("O_18.lin.weight")
        kept, m = trial("H5-blank-row-donor",
                        "the blank row's weights were altered (rel change 0.44, the largest among the output rows): row 0 <- parent row 0",
                        "delta -0.05 to -0.20 if planted; else refused (parent row misaligned to fine-tuned features)",
                        "set_rows from donor",
                        "set_rows", ["O_18.lin.weight"], "blank row from the parent",
                        index=[0], values=Wp[0].astype(np.float32))
        if kept:
            last = m

    # ---------------------------------------------------------------- H6: control, C_3.co.bias
    if left() >= 1:
        kept, m = trial("H6-C3-bias-control",
                        "control: the C_3.co.bias anomaly (rel 1.09, cos 0.68) is an absolute change of ~0.07 like the other conv biases, "
                        "i.e. fine-tuning, not a defect; transplanting the parent's bias should change nothing",
                        "|delta| < 0.01, refused by the rule",
                        "transplant from donor",
                        "transplant", ["C_3.co.bias"], "control transplant", donor=P["C_3.co.bias"].astype(np.float32))
        if kept:
            last = m

    s.note(f"end: CER {last.cer:.4f} collapsed {last.collapsed}; passes used {used}, left {left()}")


try:
    main()
except Exception as e:
    s.note(f"program error: {e!r}")
    if pending:
        try:
            s.revert(f"reverted after error: {e!r}")
        except Exception:
            pass
finally:
    summary = s.freeze()
    print(summary.get("baseline"), "->", summary.get("last_pass"))
