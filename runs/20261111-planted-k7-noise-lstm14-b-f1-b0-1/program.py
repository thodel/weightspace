"""Blind programmer (budget 0), planted-defect control, session s02 task (b).

No pass exists, so nothing here is measured. The pack (describe / diff-to-parent / C0 practice) shows no
signature that justifies an unconditional edit (hypotheses.md, H0): the practice errors are deletions that
are about what ten fully collapsed lines of 300 would contribute, the LSTM biases sit within 5 % of the parent,
every tensor's cosine to the parent is in the 0.92-0.999 band of ordinary continued training, and the space
deletions on the non-collapsed lines are balanced by space insertions.

So this program applies a repair only where a weight-only comparison against the parent (loaded as a donor;
no forward pass) finds a signature that continued training does not produce:
  H1  an output row (by character) swapped with another, replaced, or negated
  H2  an output row scaled far outside its peers (>2.2x or <0.35x the tensor's median child/parent ratio),
      or an untrained row (cos > 0.999 to the parent) whose norm ratio deviates >1.3x from the other untrained rows
  H3  the output bias of a frequent practice character shifted > 4 from the parent beyond the shared shift
  H4  a slice (row / column / conv filter / in-channel) of a conv or LSTM weight that is dead, negated, replaced
      or scaled far outside the tensor's distribution
  H5  one LSTM gate block of an input or recurrent weight rescaled by >1.5x or <0.67x relative to the other three
  H6  a whole tensor rescaled (cos > 0.999, norm ratio off by >1.2x) or negated
Each check skips a tensor when it flags more than 6 % of the slices (then my model of what is natural is wrong
and nothing is applied). If nothing fires the model is frozen unchanged.
"""
from collections import Counter

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
OW, OB = "O_18.lin.weight", "O_18.lin.bias"

MAX_FRAC = 0.06                  # more flagged slices than this share of a tensor: distrust the check, skip
HIGH, LOW, DEAD = 2.2, 0.35, 0.02  # slice norm ratio child/parent, relative to the tensor's median ratio
Z = 6.0                          # ... and at least this many robust sigmas away
NEG_COS, LOW_COS = -0.5, 0.3     # slice cosine to the parent slice: negated / unrelated
PURE_COS, PURE_DEV = 0.999, 1.3  # untrained output rows: cos to parent above, norm ratio deviation beyond
GATE_HI, GATE_LO, GATE_COS = 1.5, 0.67, 0.8
WHOLE_COS, WHOLE_DEV = 0.999, 1.2
BIAS_ABS, BIAS_Z, FREQ_MIN = 4.0, 6.0, 50
SWAP_COS = 0.8


# ----------------------------------------------------------------------------- helpers
def l2(x, axis=None):
    return np.sqrt((np.asarray(x, dtype=np.float64) ** 2).sum(axis=axis))


def others(w, axis):
    return tuple(a for a in range(w.ndim) if a != axis)


def slice_norms(w, axis):
    return l2(w, others(w, axis))


def slice_cos(a, b, axis):
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    return (a64 * b64).sum(axis=others(a64, axis)) / (slice_norms(a64, axis) * slice_norms(b64, axis) + 1e-12)


def robust(x):
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    return med, max(mad, 0.05)


def cap(n):
    return max(1, int(MAX_FRAC * n))


class Prog:
    """hypothesis -> edit -> keep_unmeasured, with the failure path closed (an edit that raises is reverted)."""

    def __init__(self, s, parent):
        self.s = s
        self.T = s.subject.tensors()      # live view: look tensors up by name after every edit
        self.P = parent.tensors()
        self.edits = 0

    def apply(self, text, prediction, cls, op, names, **kw):
        s = self.s
        h = s.hypothesis(text, prediction, cls)
        try:
            s.edit(op, names, hypothesis=h, **kw)
        except Exception as e:                       # noqa: BLE001
            s.note(f"edit failed, nothing applied: {e!r}")
            if getattr(s, "_last_edit", None) is not None:
                s.revert("edit raised")
            return False
        s.keep_unmeasured("budget 0: kept on a weight-only signature against the parent, see the hypothesis")
        self.edits += 1
        return True


# ----------------------------------------------------------------------------- generic slice checks (H4, and H2 on the output rows)
def fix_slices(p, name, axis, get_view, parent_view, to_full, label, direction=True):
    """Slices along ``axis`` of an aligned (child view, parent view) pair. ``to_full(idx, values)`` turns indices and
    values of the view into those of the full tensor. Direction anomalies first, then norm-ratio anomalies."""
    s = p.s
    view = get_view()
    n = view.shape[axis]
    if n < 4 or parent_view.shape != view.shape:
        return
    op = "set_rows" if axis == 0 else "set_cols"
    npar = slice_norms(parent_view, axis)

    if direction:
        nc = slice_norms(view, axis)
        cos = slice_cos(view, parent_view, axis)
        valid = (npar > 1e-8) & (nc > 1e-8)
        bad = [int(i) for i in np.where(valid & (cos < LOW_COS))[0]]
        if bad:
            if len(bad) > cap(n):
                s.note(f"{label}: {len(bad)} of {n} slices have cos < {LOW_COS} to the parent; too many for a blind repair, left alone")
            else:
                vals = [(-np.take(view, [i], axis=axis)) if cos[i] < NEG_COS else np.take(parent_view, [i], axis=axis) for i in bad]
                idx_full, values = to_full(bad, np.concatenate(vals, axis=axis))
                desc = "; ".join(f"{i}: cos {cos[i]:+.2f}, ratio {nc[i] / max(npar[i], 1e-8):.2f}" for i in bad)
                p.apply(f"H4 {label}: {len(bad)} slice(s) point away from the parent slice ({desc}); the other slices of this tensor "
                        f"keep cos > {LOW_COS}. A negated slice (cos < {NEG_COS}) is negated back, any other is replaced by the parent slice.",
                        "CER down, 0.1-3 pt depending on what the slice carries; training does not flip or replace single slices",
                        "direction repair: negate / parent transplant", op, [name], index=idx_full, values=values.astype(np.float32))
                view = get_view()

    nc = slice_norms(view, axis)
    ok = npar > 1e-8
    if ok.sum() < 4:
        return
    ratio = nc / np.maximum(npar, 1e-8)
    x = np.log(np.maximum(ratio, 1e-12))
    med0 = float(np.median(x[ok]))
    live = ok & (ratio > DEAD * np.exp(med0))
    med, mad = robust(x[live]) if live.sum() >= 4 else (med0, 0.05)
    hi = (x - med) > max(np.log(HIGH), Z * mad)
    lo = (med - x) > max(np.log(1.0 / LOW), Z * mad)
    dead = ratio <= DEAD * np.exp(med)
    flagged = [int(i) for i in np.where(ok & (hi | lo | dead))[0]]
    if not flagged:
        return
    if len(flagged) > cap(n):
        s.note(f"{label}: {len(flagged)} of {n} slices have a child/parent norm ratio far from the median {np.exp(med):.2f}; "
               "too many for a blind repair, left alone")
        return
    vals = []
    for i in flagged:
        if dead[i]:
            vals.append(np.take(parent_view, [i], axis=axis) * np.exp(med))
        else:
            vals.append(np.take(view, [i], axis=axis) * (np.exp(med) / ratio[i]))
    idx_full, values = to_full(flagged, np.concatenate(vals, axis=axis))
    desc = "; ".join(f"{i}: ratio {ratio[i]:.3f}" + (" (dead)" if dead[i] else "") for i in flagged)
    p.apply(f"H4 {label}: {len(flagged)} slice(s) with a child/parent norm ratio far outside the tensor's distribution "
            f"(median {np.exp(med):.2f}, robust sigma {mad:.2f} in log; {desc}). A dead slice takes the parent slice at the median "
            "ratio, a scaled slice is rescaled to the median ratio with its direction kept.",
            "CER down 0.1-3 pt if the slice was planted; a natural slice this far out is very unlikely",
            "scale repair: rescale to the median child/parent ratio (or parent transplant of a dead slice)",
            op, [name], index=idx_full, values=values.astype(np.float32))


# ----------------------------------------------------------------------------- H5, H6 on conv / LSTM tensors
def fix_whole(p, name):
    Wc, Wp = p.T[name], p.P[name]
    if Wc.shape != Wp.shape:
        return
    nc, npar = l2(Wc), l2(Wp)
    if nc < 1e-8 or npar < 1e-8:
        return
    cos = float((Wc.astype(np.float64) * Wp.astype(np.float64)).sum() / (nc * npar))
    r = nc / npar
    if cos < NEG_COS:
        p.apply(f"H6 {name}: whole tensor has cos {cos:+.3f} to the parent, i.e. it is negated (training keeps cos > 0.9)",
                "CER down several points; a negated layer is a broken layer", "negate (scale -1)", "scale", [name], factor=-1.0)
    elif cos > WHOLE_COS and abs(np.log(r)) > np.log(WHOLE_DEV):
        p.apply(f"H6 {name}: whole tensor is the parent's direction (cos {cos:.4f}) at {r:.3f}x its norm: a pure rescale, "
                "which training never produces (every trained tensor of this model sits at cos 0.92-0.98)",
                "CER down 1-5 pt", "rescale to the parent norm", "scale", [name], factor=float(1.0 / r))


def fix_gates(p, name):
    Wc, Wp = p.T[name], p.P[name]
    if Wc.ndim != 2 or Wc.shape != Wp.shape or Wc.shape[0] % 4:
        return
    H = Wc.shape[0] // 4
    blocks = [slice(g * H, (g + 1) * H) for g in range(4)]
    r = np.array([l2(Wc[b]) / max(l2(Wp[b]), 1e-8) for b in blocks])
    c = np.array([float((Wc[b].astype(np.float64) * Wp[b].astype(np.float64)).sum() / max(l2(Wc[b]) * l2(Wp[b]), 1e-12)) for b in blocks])
    flagged = []
    for g in range(4):
        rest = np.median([r[k] for k in range(4) if k != g])
        if c[g] > GATE_COS and (r[g] / rest > GATE_HI or r[g] / rest < GATE_LO):
            flagged.append((g, float(rest)))
    if len(flagged) != 1:
        if flagged:
            p.s.note(f"{name}: {len(flagged)} gate blocks out of scale with the rest ({np.round(r, 3).tolist()}); ambiguous, left alone")
        return
    g, rest = flagged[0]
    b = blocks[g]
    factor = rest / r[g]
    gate = ["input", "forget", "cell", "output"][g]
    p.apply(f"H5 {name}: the {gate}-gate block (rows {b.start}-{b.stop - 1}) has child/parent norm ratio {r[g]:.3f} against "
            f"{rest:.3f} for the other three gates, with its direction intact (cos {c[g]:.3f}); a block rescale, not training. "
            f"Rescaled by {factor:.3f}.",
            "CER down 1-4 pt (a gate out of scale saturates or silences its LSTM)", "gate block rescale (set_rows)",
            "set_rows", [name], index=list(range(b.start, b.stop)), values=(Wc[b] * factor).astype(np.float32))


# ----------------------------------------------------------------------------- H1, H2, H3 on the output layer
def output_alignment(cm, pm):
    shared = sorted(c for c in cm if c in pm)
    rows_c = np.array([0] + [cm[c] for c in shared], dtype=np.int64)
    rows_p = np.array([0] + [pm[c] for c in shared], dtype=np.int64)
    return rows_c, rows_p, ["<blank>"] + shared


def fix_output_rows_direction(p, rows_c, rows_p, names):
    s = p.s
    A = p.T[OW][rows_c].astype(np.float64)
    B = p.P[OW][rows_p].astype(np.float64)
    na, nb = l2(A, 1), l2(B, 1)
    cos = (A * B).sum(1) / (na * nb + 1e-12)
    bad = [int(i) for i in np.where((cos < LOW_COS) & (na > 1e-8) & (nb > 1e-8))[0]]
    if not bad:
        return
    if len(bad) > cap(len(names)):
        s.note(f"{OW}: {len(bad)} of {len(names)} aligned rows have cos < {LOW_COS} to their parent row; too many for a blind repair")
        return
    S = (A / (na[:, None] + 1e-12)) @ (B / (nb[:, None] + 1e-12)).T     # S[i, j]: child row of char i vs parent row of char j
    bc, bp = p.T[OB], p.P[OB]
    new_w, new_b, done, desc = {}, {}, set(), []
    for i in bad:
        if i in done:
            continue
        if cos[i] < NEG_COS:
            new_w[i] = -A[i]
            desc.append(f"'{names[i]}' negated (cos {cos[i]:+.2f})")
            continue
        j = int(np.argmax(S[i]))
        mutual = j != i and S[i, j] > SWAP_COS and int(np.argmax(S[j])) == i and S[j, i] > SWAP_COS
        if mutual:
            new_w[i], new_w[j] = A[j], A[i]
            done.add(j)
            bi, bj = float(bc[rows_c[i]]), float(bc[rows_c[j]])
            pi, pj = float(bp[rows_p[i]]), float(bp[rows_p[j]])
            if abs(bi - pj) + abs(bj - pi) < abs(bi - pi) + abs(bj - pj):
                new_b[i], new_b[j] = bj, bi
            desc.append(f"'{names[i]}' <-> '{names[j]}' swapped back (cos to each other's parent row {S[i, j]:.2f}/{S[j, i]:.2f}"
                        + (", biases too)" if i in new_b else ")"))
        else:
            new_w[i] = B[i]
            desc.append(f"'{names[i]}' replaced by the parent row (cos {cos[i]:+.2f}, best parent match {S[i, j]:.2f})")
    idx = sorted(new_w)
    per = {OW: {"index": [int(rows_c[i]) for i in idx], "values": np.stack([new_w[i] for i in idx]).astype(np.float32)}}
    tnames = [OW]
    if new_b:
        bidx = sorted(new_b)
        tnames.append(OB)
        per[OB] = {"index": [int(rows_c[i]) for i in bidx], "values": np.array([new_b[i] for i in bidx], dtype=np.float32)}
    p.apply(f"H1 output rows: {len(idx)} row(s) do not point along their parent row by character ({'; '.join(desc)}); "
            "the other aligned rows keep cos > 0.86. A mutual match with another character's parent row is a swap and is "
            "swapped back; a negated row is negated back; anything else takes the parent row.",
            "CER down 0.1-2 pt depending on how often the characters occur (a swap of two frequent letters would show as a "
            "top confusion, which the practice set does not; so the gain is small but the direction is sure)",
            "output row swap-back / negate / parent transplant (set_rows)", "set_rows", tnames, per_tensor=per)


def fix_output_rows_pure_scale(p, rows_c, rows_p, names):
    A = p.T[OW][rows_c].astype(np.float64)
    B = p.P[OW][rows_p].astype(np.float64)
    na, nb = l2(A, 1), l2(B, 1)
    cos = (A * B).sum(1) / (na * nb + 1e-12)
    untrained = (cos > PURE_COS) & (nb > 1e-8) & (na > 1e-8)
    if untrained.sum() < 10:
        return
    x = np.log(na / np.maximum(nb, 1e-8))
    med = float(np.median(x[untrained]))
    flagged = [int(i) for i in np.where(untrained & (np.abs(x - med) > np.log(PURE_DEV)))[0]]
    if not flagged:
        return
    if len(flagged) > cap(int(untrained.sum())):
        p.s.note(f"{OW}: {len(flagged)} untrained rows off the common norm ratio; too many, left alone")
        return
    values = np.stack([A[i] * np.exp(med - x[i]) for i in flagged]).astype(np.float32)
    desc = "; ".join(f"'{names[i]}' ratio {np.exp(x[i]):.2f}" for i in flagged)
    p.apply(f"H2 output rows: {len(flagged)} row(s) keep the parent's direction exactly (cos > {PURE_COS}, i.e. untouched by "
            f"training) but sit at a norm ratio far from the {int(untrained.sum())} other such rows (median {np.exp(med):.2f}; {desc}): "
            "a pure rescale. Rescaled to the common ratio.",
            "CER down 0-0.5 pt (a boosted rare row inserts its glyph; a shrunk one only loses a rare character)",
            "output row rescale (set_rows)", "set_rows", [OW], index=[int(rows_c[i]) for i in flagged], values=values)


def fix_output_bias(p, rows_c, rows_p, names, freq):
    bc = p.T[OB][rows_c].astype(np.float64)
    bp = p.P[OB][rows_p].astype(np.float64)
    d = bc - bp
    med, mad = robust(d)
    flagged = [i for i in range(1, len(names)) if freq.get(names[i], 0) >= FREQ_MIN
               and abs(d[i] - med) > max(BIAS_ABS, BIAS_Z * mad)]
    if not flagged:
        return
    if len(flagged) > cap(len(names)):
        p.s.note(f"{OB}: {len(flagged)} frequent characters with a bias shift > {BIAS_ABS} from the parent; too many, left alone")
        return
    values = np.array([bp[i] + med for i in flagged], dtype=np.float32)
    desc = "; ".join(f"'{names[i]}' bias {bc[i]:+.2f} (parent {bp[i]:+.2f}, shift {d[i] - med:+.2f} beyond the shared {med:+.2f})"
                     for i in flagged)
    p.apply(f"H3 output bias: {len(flagged)} character(s) that occur >= {FREQ_MIN} times in the practice text have a bias shifted "
            f"> {BIAS_ABS} from the parent beyond the shift shared by all rows (robust sigma {mad:.2f}; {desc}). Training moves a "
            "frequent character's bias by well under 1; rare characters are pushed down by at most ~3. Set to parent + shared shift.",
            "CER down 0.5-3 pt if it fires (a space-bias plant of this size would delete a good share of the spaces)",
            "output bias restore (set_rows)", "set_rows", [OB], index=[int(rows_c[i]) for i in flagged], values=values)


# ----------------------------------------------------------------------------- main
def main():
    s = open_session(".")
    parent = KrakenSubject(PARENT)
    p = Prog(s, parent)
    T, P = p.T, p.P
    s.note("blind programmer (budget 0, claude-fable-5-1): no unconditional edit from the pack; weight-only checks against the "
           "parent follow, each applied only on a signature continued training does not produce (program.py docstring).")
    try:
        cm = s.subject.codec_map()
        pm = parent.codec_map()
        rows_c, rows_p, names = output_alignment(cm, pm)
        freq = Counter("".join(it.get("text", "") for it in s.practice))
        s.note(f"aligned output rows: {len(names)} (incl. blank); practice characters with >= {FREQ_MIN} occurrences: "
               f"{sum(1 for c in cm if freq.get(c, 0) >= FREQ_MIN)}")

        # conv and LSTM tensors: whole-tensor, gate-block, then slice checks
        for name in sorted(T):
            if name.startswith("O_18") or name not in P or T[name].shape != P[name].shape:
                continue
            if "bias_ih" in name:
                continue                                  # all zero by construction, in both models
            fix_whole(p, name)
            if T[name].ndim == 1:
                continue
            if ".layer.weight_" in name:
                fix_gates(p, name)
            Wp = P[name]
            for axis in (0, 1):
                if T[name].shape[axis] < 4:
                    continue
                fix_slices(p, name, axis, (lambda n=name: T[n]), Wp, (lambda idx, v: (idx, v)),
                           f"{name} axis {axis}")

        # output layer, aligned by character
        fix_output_rows_direction(p, rows_c, rows_p, names)
        fix_output_rows_pure_scale(p, rows_c, rows_p, names)
        fix_slices(p, OW, 0, (lambda: T[OW][rows_c]), P[OW][rows_p],
                   (lambda idx, v: ([int(rows_c[i]) for i in idx], v)), f"{OW} rows (by character)", direction=False)

        def cols_to_full(idx, v):
            full = T[OW][:, idx].copy()
            full[rows_c] = v
            return idx, full
        fix_slices(p, OW, 1, (lambda: T[OW][rows_c]), P[OW][rows_p], cols_to_full, f"{OW} columns (over the shared rows)")
        fix_output_bias(p, rows_c, rows_p, names, freq)
    except Exception as e:                                # noqa: BLE001
        s.note(f"checks aborted: {e!r}")
        if getattr(s, "_last_edit", None) is not None:
            s.revert("aborted mid-edit")
    s.note(f"done: {p.edits} edit(s) applied unmeasured" if p.edits else
           "done: no signature fired; the model is frozen unchanged (the pack alone did not justify a blind edit)")
    s.freeze()


if __name__ == "__main__":
    main()
