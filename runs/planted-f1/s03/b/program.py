"""planted-f1 / s03 / task (b): blind programmer, budget 0.

No pass exists. Every edit below is conditional on a comparison of the subject with its parent
(kraken-german-xix-v1) that is computed inside this program from the weights alone:

  * rows / columns of every conv and LSTM weight whose norm ratio to the parent row is an outlier
    against the tensor's own median ratio (planted `scale` on rows/cols) or whose direction no longer
    matches the parent row (planted `set_rows`/`set_cols` with foreign values);
  * bias elements whose child - parent shift is an outlier against the vector's other elements;
  * output layer rows / columns (aligned by character) with the same two tests;
  * output biases of the frequent classes (space, letters, . ,) whose parent -> child shift is far
    below the letters' median shift; the blank only if its relative shift is very large.

Thresholds are set so that the natural fine-tuning drift seen in the pack (rel L2 0.12-0.33 per
tensor, cos >= 0.86 and norm ratio <= 1.36 for the most-moved output rows) never fires. If nothing
fires, the session freezes with notes only. See hypotheses.md.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

RATIO_FLOOR = float(np.log(1.75))   # a row must have grown/shrunk by >= 1.75x relative to the median ratio
Z_SCALE = 6.0                       # ... and by >= 6 robust sigma of the tensor's log ratios
COS_MAX = 0.5                       # a replaced row: cosine to parent < 0.5 ...
Z_COS = 8.0                         # ... and 8 robust sigma below the median cosine
BIAS_Z = 10.0                       # bias element: |shift - median| > max(10 robust sigma, BIAS_FLOOR)
BIAS_FLOOR = 0.3
FREQ_DROP = -3.5                    # frequent output class: relative bias shift below this is a plant
BLANK_RISE = 3.5                    # blank: relative bias shift at or above this is a plant
MAX_FRACTION = 0.30                 # more flagged rows than this: the median is no reference, leave the tensor
MAX_EDITS = 10

s = open_session(".")
T = s.subject.tensors()                     # live view
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
P = parent.tensors()
pm = parent.codec_map()
state = {"edits": 0}

s.note("blind programmer (budget 0). Readings: CER 0.0789, 14 collapsed lines, deletions only; the non-space "
       "deletion counts match German letter frequencies over ~600 chars = the 14 collapsed lines, so the other "
       "286 lines run at ~3 % CER. Tensor-level diff to the parent is uniform (rel 0.12-0.33, cos 0.95-0.99). "
       "Every edit below is conditional on an in-program outlier test against the parent; nothing is applied "
       "unconditionally.")


# ----------------------------------------------------------------------------------------------
def robust(x):
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, mad


def fmt(x):
    return f"{x:.3f}"


def row_check(C, Pm):
    """C, Pm: [n, d] float64, row-aligned. Returns None (too few rows) or a dict with
    scaled (indices, factors), replaced (indices), log_med and a one-line summary."""
    n = C.shape[0]
    nc = np.linalg.norm(C, axis=1)
    npar = np.linalg.norm(Pm, axis=1)
    med_p = float(np.median(npar))
    strong = npar >= 0.25 * med_p
    if strong.sum() < 8:
        return None
    logr = np.zeros(n)
    logr[strong] = np.log(np.maximum(nc[strong], 1e-12) / npar[strong])
    med, sig = robust(logr[strong])
    thr = max(Z_SCALE * sig, RATIO_FLOOR)
    med_c = float(np.median(nc[strong]))
    dead = strong & (nc < 0.05 * med_c)
    cos = np.ones(n)
    den = nc * npar
    ok = strong & (den > 0)
    cos[ok] = (C[ok] * Pm[ok]).sum(1) / den[ok]
    medc, sigc = robust(cos[strong])
    cos_thr = min(COS_MAX, medc - Z_COS * sigc)
    replaced = dead | (strong & (cos < cos_thr))
    weak = (~strong) & (npar >= 0.02 * med_p)
    scaled = (~replaced) & ((strong & (np.abs(logr - med) > thr)) | (weak & (nc > 1.5 * med_c)))
    sc_idx = np.flatnonzero(scaled)
    factors = np.exp(med - np.log(np.maximum(nc[sc_idx], 1e-12) / np.maximum(npar[sc_idx], 1e-12)))
    rp_idx = np.flatnonzero(replaced)
    dev = np.abs(logr[strong] - med)
    summary = (f"n={n} strong={int(strong.sum())} median ratio={fmt(np.exp(med))} sigma_log={fmt(sig)} "
               f"thr_log={fmt(thr)} max|dev|={fmt(dev.max())} (ratio {fmt(np.exp(dev.max()))}) "
               f"min cos={fmt(cos[strong].min())} median cos={fmt(medc)} cos_thr={fmt(cos_thr)} "
               f"-> scaled {len(sc_idx)}, replaced {len(rp_idx)}")
    return {"scaled": sc_idx, "factors": factors, "replaced": rp_idx, "log_med": med, "summary": summary}


def can_edit(name, n_flag, n_total):
    if state["edits"] >= MAX_EDITS:
        s.note(f"{name}: {n_flag} flagged but the edit cap ({MAX_EDITS}) is reached; left alone")
        return False
    if n_flag > MAX_FRACTION * n_total:
        s.note(f"{name}: {n_flag}/{n_total} flagged (> {MAX_FRACTION:.0%}); the median is no reference; left alone")
        return False
    return True


def repair_axis(name, axis, label):
    """Rows (axis 0) or columns (axis 1) of a tensor that exists in parent and child with the same shape."""
    c = T[name].astype(np.float64)
    p = P[name].astype(np.float64)
    if c.shape != p.shape:
        s.note(f"{name}: shape differs from parent; skipped")
        return
    if c.ndim < 2 or c.shape[axis] < 8:
        return
    if axis == 0:
        C, Pm = c.reshape(c.shape[0], -1), p.reshape(p.shape[0], -1)
    else:
        C = np.moveaxis(c, 1, 0).reshape(c.shape[1], -1)
        Pm = np.moveaxis(p, 1, 0).reshape(p.shape[1], -1)
    r = row_check(C, Pm)
    if r is None:
        return
    s.note(f"{name} {label}: {r['summary']}")
    idx = list(map(int, r["scaled"])) + list(map(int, r["replaced"]))
    if not idx:
        return
    if not can_edit(name, len(idx), C.shape[0]):
        return
    cur = T[name]
    par = P[name].astype(np.float32)
    scale_back = float(np.exp(r["log_med"]))
    if axis == 0:
        vals = [cur[i] * np.float32(f) for i, f in zip(r["scaled"], r["factors"])]
        vals += [par[i] * np.float32(scale_back) for i in r["replaced"]]
        values = np.stack(vals).astype(np.float32)
        op = "set_rows"
    else:
        vals = [cur[:, i] * np.float32(f) for i, f in zip(r["scaled"], r["factors"])]
        vals += [par[:, i] * np.float32(scale_back) for i in r["replaced"]]
        values = np.stack(vals, axis=1).astype(np.float32)
        op = "set_cols"
    what = []
    if len(r["scaled"]):
        what.append(f"{len(r['scaled'])} {label} with norm ratio to the parent an outlier "
                    f"(factors {', '.join(fmt(1 / f) for f in r['factors'][:6])}) rescaled to the tensor's median ratio")
    if len(r["replaced"]):
        what.append(f"{len(r['replaced'])} {label} with cosine to the parent far below the others (or dead) "
                    f"transplanted from the parent at the median norm ratio")
    h = s.hypothesis(
        f"{name}: {'; '.join(what)} (indices {idx[:12]}). A scaled/replaced {label[:-1]} saturates or garbles the "
        f"unit's input for some lines; this is the input-dependent defect behind the 14 collapsed lines.",
        "CER down: if this is the planted defect, collapsed lines 14 -> ~0 and CER -3 to -4.5 pp; if these were "
        "natural rows (unlikely at the thresholds), at most +0.3 pp.",
        op)
    s.edit(op, [name], hypothesis=h, index=idx, values=values)
    s.keep_unmeasured(f"budget 0: {op} on {len(idx)} {label} of {name}, outliers against the parent")
    state["edits"] += 1


def repair_bias(name):
    c = T[name].astype(np.float64)
    p = P[name].astype(np.float64)
    if c.shape != p.shape:
        return
    d = c - p
    med, sig = robust(d)
    thr = max(BIAS_Z * sig, BIAS_FLOOR)
    flagged = np.flatnonzero(np.abs(d - med) > thr)
    s.note(f"{name}: shift child-parent median={fmt(med)} sigma={fmt(sig)} thr={fmt(thr)} "
           f"max|dev|={fmt(np.abs(d - med).max())} -> {len(flagged)} flagged")
    if not len(flagged):
        return
    if not can_edit(name, len(flagged), len(d)):
        return
    delta = np.zeros_like(c)
    delta[flagged] = -(d[flagged] - med)
    h = s.hypothesis(
        f"{name}: {len(flagged)} elements (indices {flagged[:12].tolist()}) moved {', '.join(fmt(v) for v in d[flagged][:6])} "
        f"from the parent while the others moved {fmt(med)} +- {fmt(sig)}: a planted bias shift; set back to parent + median shift.",
        "CER down if planted (a shifted gate/conv bias distorts every line): -0.5 to -3 pp; if natural, within +0.2 pp.",
        "add")
    s.edit("add", [name], hypothesis=h, delta=delta.astype(np.float32))
    s.keep_unmeasured(f"budget 0: {len(flagged)} outlier elements of {name} set back to the parent's level")
    state["edits"] += 1


# ----------------------------------------------------------------------------------------------
# 1. conv and LSTM weights: rows, then columns
for name in sorted(T):
    if name.startswith("O_18"):
        continue
    if name.endswith(".weight") or ".weight_" in name:
        repair_axis(name, 0, "rows")
        repair_axis(name, 1, "columns")

# 2. conv and LSTM biases, element-wise
for name in sorted(T):
    if name.startswith("O_18"):
        continue
    if ".bias" in name:
        repair_bias(name)

# 3. output layer, rows aligned by character (blank = row 0 in both)
WN, BN = "O_18.lin.weight", "O_18.lin.bias"
shared = sorted(ch for ch in cm if ch in pm)
ci = np.array([0] + [cm[ch] for ch in shared])
pi = np.array([0] + [pm[ch] for ch in shared])
s.note(f"output layer: {len(shared)} shared characters + blank aligned; child rows {T[WN].shape[0]}, parent {P[WN].shape[0]}")

Wc = T[WN].astype(np.float64)[ci]
Wp = P[WN].astype(np.float64)[pi]
r = row_check(Wc, Wp)
if r is not None:
    s.note(f"{WN} rows (aligned): {r['summary']}")
    k_idx = list(map(int, r["scaled"])) + list(map(int, r["replaced"]))
    if k_idx and can_edit(WN, len(k_idx), len(ci)):
        rows = [int(ci[k]) for k in k_idx]
        vals = [T[WN][ci[k]] * np.float32(f) for k, f in zip(r["scaled"], r["factors"])]
        vals += [P[WN][pi[k]].astype(np.float32) * np.float32(np.exp(r["log_med"])) for k in r["replaced"]]
        chars = ["<blank>" if k == 0 else shared[k - 1] for k in k_idx]
        h = s.hypothesis(
            f"{WN}: output rows {chars[:12]} are norm/direction outliers against their parent rows; rescale / transplant.",
            "CER down by the share of those characters' errors (-0.2 to -2 pp); natural rows would cost <= +0.2 pp.",
            "set_rows")
        s.edit("set_rows", [WN], hypothesis=h, index=rows, values=np.stack(vals).astype(np.float32))
        s.keep_unmeasured("budget 0: output rows that are outliers against the parent, aligned by character")
        state["edits"] += 1

# 4. output layer columns (features), on the aligned rows
Wc = T[WN].astype(np.float64)[ci]
r = row_check(Wc.T, Wp.T)
if r is not None:
    s.note(f"{WN} columns (aligned rows): {r['summary']}")
    cols = list(map(int, r["scaled"])) + list(map(int, r["replaced"]))
    if cols and can_edit(WN, len(cols), Wc.shape[1]):
        cur = T[WN]
        vals = [cur[:, j] * np.float32(f) for j, f in zip(r["scaled"], r["factors"])]
        for j in r["replaced"]:
            v = cur[:, j].copy()
            v[ci] = P[WN][pi, j].astype(np.float32) * np.float32(np.exp(r["log_med"]))
            vals.append(v)
        h = s.hypothesis(
            f"{WN}: feature columns {cols[:12]} are norm/direction outliers against the parent on the aligned rows; "
            f"rescale / transplant (new-class rows keep their values).",
            "CER down if planted (a scaled feature column shifts every frame's logits): -0.5 to -3 pp; natural <= +0.2 pp.",
            "set_cols")
        s.edit("set_cols", [WN], hypothesis=h, index=cols, values=np.stack(vals, axis=1).astype(np.float32))
        s.keep_unmeasured("budget 0: output columns that are outliers against the parent")
        state["edits"] += 1

# 5. output biases: frequent classes relative to the letters' median shift; blank; rare classes pushed far down
bc = T[BN].astype(np.float64)
bp = P[BN].astype(np.float64)
letters = [ch for ch in "enirstadhulcgmobfkwzvp" if ch in cm and ch in pm]
tested = [ch for ch in [" ", ".", ","] if ch in cm and ch in pm] + letters
shift = {ch: float(bc[cm[ch]] - bp[pm[ch]]) for ch in shared}
blank_shift = float(bc[0] - bp[0])
delta = np.zeros_like(bc)
reasons = []
if len(letters) >= 8:
    for ch in tested:
        ref = np.median([shift[o] for o in letters if o != ch])
        dev = shift[ch] - ref
        if dev < FREQ_DROP:
            fix = min(-dev, 8.0)
            delta[cm[ch]] += fix
            reasons.append(f"{ch!r}: shift {fmt(shift[ch])} vs letters' median {fmt(ref)} (dev {fmt(dev)}) -> +{fmt(fix)}")
    ref_all = float(np.median([shift[o] for o in letters]))
    blank_dev = blank_shift - ref_all
    s.note(f"{BN}: blank shift {fmt(blank_shift)}, letters' median shift {fmt(ref_all)} (blank relative {fmt(blank_dev)}); "
           f"space shift {fmt(shift.get(' ', float('nan')))}, space bias child {fmt(bc[cm[' ']]) if ' ' in cm else 'n/a'}; "
           f"frequent-class deviations: " + ", ".join(f"{ch!r}:{fmt(shift[ch] - ref_all)}" for ch in tested))
    if blank_dev >= BLANK_RISE:
        fix = -(blank_dev - 1.0)
        delta[0] += fix
        reasons.append(f"blank: shift {fmt(blank_shift)} vs letters {fmt(ref_all)} (dev {fmt(blank_dev)}) -> {fmt(fix)}")
    # rare / other shared classes pushed far down against the shared median (downward only)
    others = [ch for ch in shared if ch not in tested]
    if len(others) >= 20:
        vals = np.array([shift[ch] for ch in others])
        med_o, sig_o = robust(vals)
        thr_o = max(BIAS_Z * sig_o, 4.0)
        for ch, v in zip(others, vals):
            if v - med_o < -thr_o:
                fix = min(-(v - med_o), 8.0)
                delta[cm[ch]] += fix
                reasons.append(f"{ch!r}: shift {fmt(v)} vs others' median {fmt(med_o)} -> +{fmt(fix)}")
        s.note(f"{BN}: other shared classes shift median {fmt(med_o)} sigma {fmt(sig_o)} thr {fmt(thr_o)} "
               f"min {fmt(vals.min())} max {fmt(vals.max())}")
n_fix = int((delta != 0).sum())
if n_fix and can_edit(BN, n_fix, len(bc)):
    h = s.hypothesis(
        f"{BN}: {n_fix} class biases moved from the parent far outside the shift of comparable classes: "
        + "; ".join(reasons[:8]) + ". Planted bias shifts; set back toward the comparable classes' shift.",
        "CER down: a lowered frequent class (space) recovers ~100 of its deletions (-0.5 to -0.8 pp); a raised blank "
        "recovers deletions in all lines (-1 to -3 pp); natural values would cost <= +0.3 pp.",
        "add")
    s.edit("add", [BN], hypothesis=h, delta=delta.astype(np.float32))
    s.keep_unmeasured("budget 0: output bias outliers against the parent, relative to comparable classes")
    state["edits"] += 1
else:
    s.note(f"{BN}: no bias outlier (space/blank/frequent classes within the natural drift); untouched")

s.note(f"done: {state['edits']} edit(s) applied, all conditional on the parent comparison")
summary = s.freeze()
print("edits:", state["edits"], "final digest:", summary.get("final_digest"))
