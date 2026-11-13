"""planted-f1 / s03 / task (a), budget 10 — the programmer's script, written in full before the first pass.

Readings (pack): deletion-only top confusions, 14 collapsed lines, length ratio 0.941; blank bias +3.23 (Δ vs parent
+2.11); space bias −7.85, the most negative of 297 rows; C_3.co.bias cosine 0.68 vs the parent; L_16.weight_ih_l0 abs
max 4.7. Plan (see preregistration_hypotheses.md): C0, H1 blank-bias reset, H2 suppressed frequent rows (space),
H3 scrambled output rows, H4 localized hidden-tensor edits (rows, cols, whole tensors), H6 blank fine step, H5
C_3.co.bias transplant, H6 second step. Every edit: hypothesis → edit → measure → keep (harness rule) with an own
check on collapsed lines. Runs inside the sandbox: wsp + numpy only.
"""
from collections import Counter

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"
NOTHING = {"", "∅", None}
MAX_SLICES = 40          # H4: at most this many slices in one repair edit (the most severe first)

s = open_session(".")
T = s.subject.tensors()                      # live view; re-indexed after every edit
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)               # donor only; never measured
PT = parent.tensors()
pm = parent.codec_map()
c2p = codec_row_map(cm, pm)                  # child row -> parent row for shared characters


# ----------------------------------------------------------------------------- metric helpers
def confusion_items(m):
    """(ref, hyp, count) triples from m.confusions, whatever container the harness uses; [] if unreadable."""
    conf = getattr(m, "confusions", None)
    out = []
    try:
        if isinstance(conf, dict):
            for k, v in conf.items():
                if isinstance(k, tuple) and len(k) == 2:
                    out.append((k[0], k[1], int(v)))
                elif isinstance(v, dict):
                    for h, n in v.items():
                        out.append((k, h, int(n)))
        else:
            for it in conf or []:
                if isinstance(it, dict):
                    r = it.get("ref", it.get("reference"))
                    h = it.get("hyp", it.get("hypothesis"))
                    out.append((r, h, int(it.get("count", it.get("n", 1)))))
                elif len(it) == 3:
                    out.append((it[0], it[1], int(it[2])))
                elif len(it) == 2 and isinstance(it[0], (tuple, list)):
                    out.append((it[0][0], it[0][1], int(it[1])))
    except Exception:
        return []
    return out


def del_ins(m):
    items = confusion_items(m)
    if not items:
        return None
    d = sum(n for r, h, n in items if h in NOTHING and r not in NOTHING)
    i = sum(n for r, h, n in items if r in NOTHING and h not in NOTHING)
    return d, i


def length_ratio(m):
    lr = getattr(m, "length_ratio", None)
    if lr is None:
        try:
            lr = m.as_dict().get("length_ratio")
        except Exception:
            lr = None
    return lr


def deletion_heavy(m):
    lr = length_ratio(m)
    if lr is not None:
        return lr < 0.985
    di = del_ins(m)
    return bool(di) and di[0] > 1.5 * di[1]


def insertion_heavy(m):
    lr = length_ratio(m)
    if lr is not None:
        return lr > 1.015
    di = del_ins(m)
    return bool(di) and di[1] > 1.5 * di[0]


def show(label, m):
    items = sorted(confusion_items(m), key=lambda t: -t[2])[:6]
    top = ", ".join(f"{r!r}->{h!r} {n}" for r, h, n in items)
    di = del_ins(m)
    txt = (f"{label}: CER {m.cer:.4f} WER {m.wer:.4f} collapsed {m.collapsed} length ratio {length_ratio(m)} "
           f"del/ins {di} | {top}")
    print(txt)
    s.note(txt)


# ----------------------------------------------------------------------------- the edit wrapper
state = {"cer": None, "collapsed": None, "last": None}       # the last kept state


def attempt(text, prediction, edit_class, op, names, label, per_tensor=None, max_collapsed_rise=3, **kwargs):
    """hypothesis -> edit -> measure -> keep / revert. Returns (kept, metrics); (None, None) when no pass is left."""
    if s.budget_left <= 0:
        s.note(f"skip {label}: no pass left")
        return None, None
    h = s.hypothesis(text, prediction, edit_class)
    pre = {n: T[n].copy() for n in names}
    try:
        s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kwargs)
    except Exception as e:                       # the op refused: nothing recorded, restore any half-applied tensor
        for n, v in pre.items():
            s.subject.set_tensor(n, v)
        s.note(f"skip {label}: edit failed ({e})")
        return None, None
    try:
        m = s.measure(label)
    except Exception as e:                       # a pass that gave nothing is not counted; undo and go on
        s.revert(f"{label}: measure failed ({e})")
        return None, None
    show(label, m)
    if state["collapsed"] is not None and m.collapsed > state["collapsed"] + max_collapsed_rise:
        s.revert(f"{label}: collapsed lines rose {state['collapsed']} -> {m.collapsed}")
        return False, m
    kept = s.keep(f"{label}: CER {state['cer']:.4f} -> {m.cer:.4f}")
    if kept:
        state.update(cer=m.cer, collapsed=m.collapsed, last=m)
    s.note(f"{label}: {'kept' if kept else 'refused/reverted'}")
    return kept, m


# ----------------------------------------------------------------------------- pass 1: baseline
m0 = s.measure("C0")
show("C0", m0)
state.update(cer=m0.cer, collapsed=m0.collapsed, last=m0)

# ----------------------------------------------------------------------------- diagnostics (no passes)
freq = Counter("".join(it["text"] for it in s.practice))
shared = [c for c in cm if c in pm]
frequent = [c for c, _ in freq.most_common() if c in cm and c in pm][:40]
dbias = {c: float(T[B][cm[c]] - PT[B][pm[c]]) for c in shared}
drift = float(np.median([dbias[c] for c in frequent]))          # natural bias drift of well-trained rows
blank_rel = float(T[B][0] - PT[B][0]) - drift


def row_rel(c):
    a, b = T[W][cm[c]].astype(np.float64), PT[W][pm[c]].astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))


wrel = {c: row_rel(c) for c in frequent}
wrel_med = float(np.median(list(wrel.values())))
suppressed = [c for c in frequent if dbias[c] - drift < -2.5]
promoted = [c for c in frequent if dbias[c] - drift > 2.5]
scrambled = [c for c in frequent if wrel[c] > max(0.6, 3 * wrel_med)]

s.note(f"diagnostics: drift(frequent bias Δ median) {drift:+.3f}; blank Δ {float(T[B][0] - PT[B][0]):+.3f}, "
       f"relative {blank_rel:+.3f}; space bias child {float(T[B][cm[' ']]) if ' ' in cm else None} "
       f"parent {float(PT[B][pm[' ']]) if ' ' in pm else None}; suppressed {suppressed!r}; promoted {promoted!r}; "
       f"scrambled rows {scrambled!r} (median row rel {wrel_med:.3f}); "
       f"frequent bias Δ-drift: " + ", ".join(f"{c!r}:{dbias[c] - drift:+.2f}" for c in frequent[:15]))


def slice_view(A, axis):
    return np.moveaxis(A.astype(np.float64), axis, 0).reshape(A.shape[axis], -1)


def outlier_slices(C, P, axis):
    """Indices along `axis` whose slice differs from the parent far more than the tensor's typical drift."""
    c, p = slice_view(C, axis), slice_view(P, axis)
    pn, cn = np.linalg.norm(p, axis=1), np.linalg.norm(c, axis=1)
    live = pn > 1e-6
    if live.sum() < 8:
        return [], np.zeros(len(pn))
    rel = np.linalg.norm(c - p, axis=1) / (pn + 1e-8)
    ratio = cn / (pn + 1e-8)
    med = np.median(rel[live])
    mad = np.median(np.abs(rel[live] - med)) + 1e-6
    big = pn > 0.2 * np.median(pn[live])
    flag = live & ((rel > max(0.75, med + 8 * mad)) | (big & ((ratio > 2.5) | (ratio < 0.4))))
    return [int(i) for i in np.flatnonzero(flag)], rel


def outlier_elements(C, P):
    a = np.abs(C.astype(np.float64) - P.astype(np.float64))
    if a.size < 8 or not np.any(a > 0):
        return [], a
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-6
    return [int(i) for i in np.flatnonzero(a > max(0.3, med + 10 * mad))], a


rows_fix, cols_fix, whole_fix, severity = {}, {}, [], {}
for name in T:
    if name.startswith("O_18") or name not in PT or PT[name].shape != T[name].shape:
        continue
    C, P = T[name], PT[name]
    if C.ndim == 1:
        idx, sev = outlier_elements(C, P)
        if idx:
            if len(idx) > 0.3 * C.shape[0]:
                whole_fix.append(name)
            else:
                rows_fix[name] = idx
                severity.update({(name, 0, i): float(sev[i]) for i in idx})
        continue
    for axis, store in ((0, rows_fix), (1, cols_fix)):
        idx, sev = outlier_slices(C, P, axis)
        if not idx:
            continue
        if len(idx) > 0.3 * C.shape[axis]:
            if name not in whole_fix:
                whole_fix.append(name)
            continue
        store[name] = idx
        severity.update({(name, axis, i): float(sev[i]) for i in idx})


def cap(store, axis):
    """Keep at most MAX_SLICES slices over all tensors of one axis, the most severe first."""
    keys = sorted(((n, i) for n, idx in store.items() for i in idx), key=lambda k: -severity.get((k[0], axis, k[1]), 0))
    keep = set(keys[:MAX_SLICES])
    return {n: [i for i in idx if (n, i) in keep] for n, idx in store.items() if any((n, i) in keep for i in idx)}


rows_fix, cols_fix = cap(rows_fix, 0), cap(cols_fix, 1)
s.note(f"hidden detector: rows {{{', '.join(f'{n}: {len(i)}' for n, i in rows_fix.items())}}}; "
       f"cols {{{', '.join(f'{n}: {len(i)}' for n, i in cols_fix.items())}}}; whole {whole_fix!r}")

# ----------------------------------------------------------------------------- H1: blank bias reset
if blank_rel > 0.5:
    target = np.array([PT[B][0] + drift], dtype=np.float32)
    attempt(f"H1 the blank bias was raised by {blank_rel:+.2f} relative to the frequent rows' drift; the deletion-only "
            f"confusions, 14 collapsed lines and length ratio 0.941 are blank over-prediction",
            f"CER down 0.020-0.035, collapsed lines to <= 5, length ratio to 0.98-1.00; refused -> blank not planted",
            "bias reset (set_rows from parent + drift)", "set_rows", [B], "H1 blank bias reset",
            index=[0], values=target)
else:
    s.note(f"H1 skipped: blank shift relative to drift is {blank_rel:+.3f} (<= 0.5)")

# ----------------------------------------------------------------------------- H2: suppressed frequent rows (space)
if suppressed:
    idx = [cm[c] for c in suppressed]
    vals = np.array([PT[B][pm[c]] + drift for c in suppressed], dtype=np.float32)
    attempt(f"H2 the bias of {suppressed!r} sits > 2.5 below parent + drift (space -7.85 is the most negative row); "
            f"their deletions (space 177) are a planted suppression",
            "CER down 0.005-0.012, WER down >= 0.05, space deletions 177 -> < 60; refused -> own calibration",
            "bias reset of frequent rows (set_rows from parent + drift)", "set_rows", [B], "H2 suppressed rows",
            index=idx, values=vals)
else:
    s.note("H2 skipped: no frequent row sits > 2.5 below parent + drift")

# ----------------------------------------------------------------------------- H3: scrambled output rows
if scrambled:
    attempt(f"H3 output rows of {scrambled!r} differ from the parent by rel {[round(wrel[c], 2) for c in scrambled]} "
            f"(median {wrel_med:.2f}); they were replaced or scrambled",
            "CER down 0.002-0.010; refused -> parent rows read worse in the drifted feature space, rows not planted",
            "row transplant from parent (set_rows weights + bias)", "set_rows", [W, B], "H3 scrambled rows",
            per_tensor={W: {"index": [cm[c] for c in scrambled], "values": PT[W][[pm[c] for c in scrambled]]},
                        B: {"index": [cm[c] for c in scrambled],
                            "values": np.array([PT[B][pm[c]] + drift for c in scrambled], dtype=np.float32)}})
else:
    s.note("H3 skipped: no frequent output row is an outlier against the parent")

# ----------------------------------------------------------------------------- H4: localized hidden-tensor edits
if rows_fix:
    attempt(f"H4a rows/units {rows_fix!r} differ from the parent far beyond their tensor's drift: a localized plant",
            "CER down 0.003-0.03, collapsed lines down; refused -> natural drift (detector false positives)",
            "row transplant from parent (set_rows)", "set_rows", list(rows_fix), "H4a hidden rows",
            per_tensor={n: {"index": idx, "values": PT[n][np.asarray(idx)]} for n, idx in rows_fix.items()})
else:
    s.note("H4a skipped: no hidden row outliers")

if cols_fix:
    attempt(f"H4b columns/features {cols_fix!r} differ from the parent far beyond their tensor's drift: a localized plant",
            "CER down 0.003-0.03, collapsed lines down; refused -> natural drift (detector false positives)",
            "column transplant from parent (set_cols)", "set_cols", list(cols_fix), "H4b hidden cols",
            per_tensor={n: {"index": idx, "values": PT[n][:, np.asarray(idx)]} for n, idx in cols_fix.items()})
else:
    s.note("H4b skipped: no hidden column outliers")

if whole_fix:
    attempt(f"H4c tensors {whole_fix!r} have > 30 % outlier slices against the parent: a whole-tensor plant (noise/scale)",
            "CER down 0.003-0.03; refused -> the tensor's drift is natural",
            "whole-tensor transplant from parent", "transplant", list(whole_fix), "H4c hidden whole tensors",
            per_tensor={n: {"donor": PT[n]} for n in whole_fix})
else:
    s.note("H4c skipped: no whole-tensor outliers")


# ----------------------------------------------------------------------------- H6: blank fine steps
def blank_step(label):
    last = state["last"]
    if deletion_heavy(last):
        step = -0.75
    elif insertion_heavy(last):
        step = +0.5
    else:
        s.note(f"{label} skipped: last kept state is balanced (length ratio {length_ratio(last)}, del/ins {del_ins(last)})")
        return None
    delta = np.zeros_like(T[B])
    delta[0] = step
    return attempt(f"{label} the kept state is still {'deletion' if step < 0 else 'insertion'}-heavy "
                   f"(length ratio {length_ratio(last)}, del/ins {del_ins(last)}): the blank bias is off by a residual",
                   f"CER down 0.002-0.008 with blank bias {step:+.2f}; refused -> blank already at its optimum",
                   "bias add", "add", [B], label, delta=delta)


blank_step("H6 blank step 1")

# ----------------------------------------------------------------------------- H5: C_3.co.bias transplant
if "C_3.co.bias" not in whole_fix:
    attempt("H5 C_3.co.bias moved with cosine 0.68 / rel 1.09 against the parent (neighbours 0.15-0.33, own weights 0.17): "
            "added noise on a tiny tensor",
            "CER down 0.000-0.005, probably within the interval; refused -> natural drift of a tiny tensor",
            "whole-tensor transplant from parent", "transplant", ["C_3.co.bias"], "H5 C_3 bias transplant",
            donor=PT["C_3.co.bias"])
else:
    s.note("H5 covered by H4c")

blank_step("H6 blank step 2")

summary = s.freeze()
print({k: summary.get(k) for k in ("used", "edits", "kept", "baseline", "last_pass")})
