"""planted-f1 / s02 / task (a): programmer with 10 passes. Written in full before the first pass.

Plan (preregistration_hypotheses.md):
  pass 1     C0 baseline.
  0 passes   structured diff of the subject against its parent (kraken-german-xix-v1), looking for the signatures a
             weight-space plant leaves and training drift does not: exact zeros, replaced/flipped/scaled slices, spikes,
             scaled gate / direction blocks, bias outliers.
  <= 5       one edit per signature class that is flagged (masked ``add`` restoring or rescaling from the parent), each
             measured, checked for collapsed lines, decided by s.keep.
  <= 3       calibration probes independent of a plant: space bias +0.75, blank bias -0.4, child-only rows -1.5.
  end        s.freeze().
Only wsp + numpy. No uncounted forward (logits / activations are not used); the practice references are not read.
"""
from __future__ import annotations

import traceback

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

SLICE_RATIO = float(np.log(1.6))    # a row / column / channel whose norm ratio to the parent leaves [1/1.6, 1.6]
BLOCK_RATIO = float(np.log(1.25))   # a gate block, a direction half, a whole weight tensor
COS_REPLACED = 0.3                  # a slice that no longer points where the parent's does
COS_PURE_SCALE = 0.9                # above this a norm change is treated as a scale and undone by rescaling
SPIKE_RMS = 8.0                     # |w| > 8 rms of its tensor ...
SPIKE_MOVE = 4.0                    # ... and moved more than 4 rms away from the parent's value
BIAS_ALL, BIAS_FREQ = 3.0, 2.0      # O_18 bias drift outlier thresholds (all rows / frequent rows)
FREQUENT = "abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIJKLMNOPQRSTUVWZ.,- "
MAX_STRUCTURAL = 5
MAX_COLLAPSE_RISE = 2

s = open_session(".")
T = s.subject.tensors()                       # live dict: always index T[name] afresh after a decision
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
P = parent.tensors()
pm = parent.codec_map()
rowmap = codec_row_map(cm, pm)                # child O_18 row -> parent row, shared characters
rowmap[0] = 0                                 # the CTC blank
char_of = {r: c for c, r in cm.items()}
char_of[0] = "<blank>"

state = {"cer": None, "wer": None, "collapsed": None}
kept_log: list[str] = []


# ---------------------------------------------------------------- helpers
def top_conf(m, k: int = 8) -> str:
    c = getattr(m, "confusions", None)
    try:
        if isinstance(c, dict):
            items = sorted(c.items(), key=lambda kv: -kv[1])[:k]
        else:
            items = list(c)[:k]
        return "; ".join(str(x) for x in items)
    except Exception as e:                    # the shape of .confusions is not documented in the brief
        return f"<confusions unreadable: {e}>"


def pair(name: str):
    """(child, parent-in-child-row-order, rows-that-have-a-parent) as float64, or None if shapes differ."""
    t = T[name].astype(np.float64)
    if name.startswith("O_18"):
        pw = P[name].astype(np.float64)
        p = np.zeros_like(t)
        has = np.zeros(t.shape[0], bool)
        for ic, ip in rowmap.items():
            if ic < t.shape[0] and ip < pw.shape[0]:
                p[ic] = pw[ip]
                has[ic] = True
        return t, p, has
    p = P.get(name)
    if p is None or p.shape != t.shape:
        return None
    return t, p.astype(np.float64), np.ones(t.shape[0], bool)


def bcast(has: np.ndarray, ndim: int) -> np.ndarray:
    return has.reshape((-1,) + (1,) * (ndim - 1))


def slice_specs(shape, has):
    """[(label, extract(array)->(n_slices x d), writeback(A, base)->array, ok_d)]: the slices of one tensor."""
    if len(shape) == 4:
        o, i, kh, kw = shape
        return [("out-channel", lambda a: a.reshape(o, -1), lambda A, base: A.reshape(shape), np.ones(i * kh * kw, bool)),
                ("in-channel", lambda a: a.transpose(1, 0, 2, 3).reshape(i, -1),
                 lambda A, base: A.reshape(i, o, kh, kw).transpose(1, 0, 2, 3), np.ones(o * kh * kw, bool))]
    if len(shape) == 2:
        r, c = shape
        idx = np.where(has)[0]

        def wb_rows(A, base, idx=idx):
            out = base.copy()
            out[idx] = A
            return out
        return [("row", lambda a: a[idx], wb_rows, np.ones(c, bool)),
                ("column", lambda a: a.T, lambda A, base: A.T, has.copy())]
    return []


def fix_slices(work: np.ndarray, p: np.ndarray, has: np.ndarray):
    """Rows / columns / channels replaced, flipped or scaled against the parent: undo. Returns (work, changes, notes)."""
    changes, notes = [], []
    for label, ext, wb, ok_d in slice_specs(work.shape, has):
        A = np.array(ext(work), dtype=np.float64, copy=True)
        B = np.array(ext(p), dtype=np.float64, copy=True)
        a, b = A[:, ok_d], B[:, ok_d]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        if not (nb > 0).any():
            continue
        typical = float(np.median(nb[nb > 0]))
        valid = nb > 0.05 * typical
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = (a * b).sum(1) / (na * nb + 1e-12)
            r = na / (nb + 1e-12)
            lr = np.abs(np.log(r + 1e-12))
        replaced = valid & (na > 0.05 * typical) & (cos < COS_REPLACED)
        scaled = valid & ~replaced & (lr > SLICE_RATIO)
        flagged = replaced | scaled
        n = int(flagged.sum())
        if n == 0:
            continue
        if n > 0.25 * len(flagged):
            notes.append(f"{label}: {n}/{len(flagged)} slices off against the parent: too many for a slice repair, "
                         f"left to the block rule")
            continue
        for i in np.where(flagged)[0]:
            if replaced[i] or r[i] < 0.1 or cos[i] < COS_PURE_SCALE:
                A[i, ok_d] = B[i, ok_d]
                kind = "transplanted"
            else:
                A[i] = A[i] / r[i]
                kind = f"rescaled x{1 / r[i]:.2f}"
            changes.append(f"{label} {int(i)}: ratio {r[i]:.2f} cos {cos[i]:.2f} -> {kind}")
        work = wb(A, work)
    return work, changes, notes


def block_list(name: str, shape):
    bl = []
    if ".layer.weight_" in name:
        u = shape[0] // 4
        for g, gn in enumerate("ifgo"):
            bl.append((f"gate {gn} rows", (slice(g * u, (g + 1) * u), slice(None))))
        if "weight_ih" in name and shape[1] == 2 * u:          # input = fwd | rev halves of the layer below
            bl.append(("input columns fwd half", (slice(None), slice(0, u))))
            bl.append(("input columns rev half", (slice(None), slice(u, 2 * u))))
    if name == "O_18.lin.weight":
        h = shape[1] // 2
        bl.append(("columns fwd half", (slice(None), slice(0, h))))
        bl.append(("columns rev half", (slice(None), slice(h, 2 * h))))
    bl.append(("whole tensor", tuple(slice(None) for _ in shape)))
    return bl


def fix_blocks(name: str, work: np.ndarray, p: np.ndarray, has: np.ndarray):
    work = work.copy()
    changes = []
    for label, sl in block_list(name, work.shape):
        a, b = work[sl], p[sl]
        hs = has[sl[0]]
        a_s, b_s = a[hs], b[hs]
        na, nb = float(np.linalg.norm(a_s)), float(np.linalg.norm(b_s))
        if nb == 0 or na == 0:
            continue
        r = na / nb
        cos = float((a_s * b_s).sum() / (na * nb))
        if abs(np.log(r)) > BLOCK_RATIO:
            if cos > COS_PURE_SCALE and r > 0.1:
                work[sl] = a / r
                changes.append(f"{label}: ratio {r:.3f} cos {cos:.3f} -> rescaled x{1 / r:.3f}")
            else:
                a2 = a.copy()
                a2[hs] = b[hs]
                work[sl] = a2
                changes.append(f"{label}: ratio {r:.3f} cos {cos:.3f} -> transplanted from the parent")
    return work, changes


def fix_bias(name: str, t: np.ndarray, p: np.ndarray, has: np.ndarray):
    d = t - p
    delta = np.zeros_like(t)
    changes = []
    if name == "O_18.lin.bias":
        if not has.any():
            return delta, changes
        med_all = float(np.median(d[has]))
        freq = np.zeros(len(t), bool)
        freq[0] = True
        for ch in FREQUENT:
            if ch in cm and ch in pm and cm[ch] < len(t):
                freq[cm[ch]] = True
        med_f = float(np.median(d[freq])) if freq.any() else med_all
        sp = cm.get(" ")
        s.note(f"O_18 bias drift vs parent: median all {med_all:+.2f}, frequent {med_f:+.2f}; blank {d[0]:+.2f}; "
               + (f"space bias {t[sp]:+.2f} parent {p[sp]:+.2f} drift {d[sp]:+.2f}" if sp is not None and has[sp] else "space has no parent row")
               + "; frequent-row drifts: " + ", ".join(f"{char_of.get(i, i)!r} {d[i]:+.2f}" for i in np.where(freq)[0][:40]))
        for i in np.where(has)[0]:
            ref = med_f if freq[i] else med_all
            thr = BIAS_FREQ if freq[i] else BIAS_ALL
            if abs(d[i] - ref) > thr:
                delta[i] = -(d[i] - ref)
                changes.append(f"row {int(i)} {char_of.get(int(i), '?')!r}: bias {t[i]:+.2f} parent {p[i]:+.2f} drift "
                               f"{d[i]:+.2f} vs median {ref:+.2f} -> {t[i] + delta[i]:+.2f}")
    else:
        med = float(np.median(d))
        mad = 1.4826 * float(np.median(np.abs(d - med)))
        thr = max(0.3, 8 * mad)
        for i in np.where(np.abs(d - med) > thr)[0]:
            delta[i] = -(d[i] - med)
            changes.append(f"{name}[{int(i)}]: {t[i]:+.3f} parent {p[i]:+.3f} drift {d[i]:+.3f} vs median {med:+.3f} (thr {thr:.3f})")
    return delta, changes


# ------------------------------------------------- the five signature classes (computed from the current weights)
def compute_zeros():
    deltas, descs = {}, []
    for name in list(T):
        if "bias_ih" in name:
            continue
        pr = pair(name)
        if pr is None:
            continue
        t, p, has = pr
        mask = (t == 0) & (p != 0) & bcast(has, t.ndim)
        n = int(mask.sum())
        if n:
            deltas[name] = (p - t) * mask
            descs.append(f"{name}: {n} exact zeros where the parent is non-zero")
    return deltas, descs


def compute_slices():
    deltas, descs = {}, []
    for name in list(T):
        if T[name].ndim < 2:
            continue
        pr = pair(name)
        if pr is None:
            continue
        t, p, has = pr
        work, changes, notes = fix_slices(t.copy(), p, has)
        for x in notes:
            s.note(f"slices {name}: {x}")
        if changes and not np.array_equal(work, t):
            deltas[name] = work - t
            descs += [f"{name} {x}" for x in changes]
    return deltas, descs


def compute_spikes():
    deltas, descs = {}, []
    for name in list(T):
        if T[name].ndim < 2:
            continue
        pr = pair(name)
        if pr is None:
            continue
        t, p, has = pr
        hb = bcast(has, t.ndim)
        tv = t[has] if not has.all() else t
        rms = float(np.sqrt(np.mean(tv ** 2)))
        if rms == 0:
            continue
        big = (np.abs(t) > SPIKE_RMS * rms) & hb
        moved = np.abs(t - p) > SPIKE_MOVE * rms
        mask = big & moved
        n_big, n_mask = int(big.sum()), int(mask.sum())
        n_pbig = int(((np.abs(p) > SPIKE_RMS * rms) & hb).sum())
        s.note(f"spikes {name}: rms {rms:.4f}; |w|>{SPIKE_RMS:.0f} rms: child {n_big}, parent {n_pbig}; of the child's, "
               f"moved >{SPIKE_MOVE:.0f} rms from the parent: {n_mask}; abs max child {np.abs(t).max():.3f} parent {np.abs(p).max():.3f}")
        if 0 < n_mask <= 0.01 * t.size:
            deltas[name] = (p - t) * mask
            where = np.argwhere(mask)[:5]
            descs.append(f"{name}: {n_mask} spikes, e.g. " + ", ".join(
                f"{tuple(int(k) for k in w)} {t[tuple(w)]:+.2f}->{p[tuple(w)]:+.2f}" for w in where))
    return deltas, descs


def compute_blocks():
    deltas, descs = {}, []
    for name in list(T):
        if T[name].ndim < 2:
            continue
        pr = pair(name)
        if pr is None:
            continue
        t, p, has = pr
        work, changes = fix_blocks(name, t, p, has)
        if changes:
            deltas[name] = work - t
            descs += [f"{name} {x}" for x in changes]
    return deltas, descs


def compute_bias():
    deltas, descs = {}, []
    for name in list(T):
        if T[name].ndim != 1 or "bias_ih" in name:
            continue
        pr = pair(name)
        if pr is None:
            continue
        t, p, has = pr
        delta, changes = fix_bias(name, t, p, has)
        if changes and np.any(delta != 0):
            deltas[name] = delta
            descs += changes
    return deltas, descs


COMPUTE = {"zeros": compute_zeros, "slices": compute_slices, "spikes": compute_spikes,
           "blocks": compute_blocks, "bias": compute_bias}

HYP = {
    "zeros": ("H1 zeroed entries: exact zeros where the parent is non-zero are a plant (training never writes exact zeros); "
              "restore the parent's values there.",
              "CER falls by 0.5-3 pp and collapsed lines fall; if the rule refuses, the zeros are inert and not the defect"),
    "slices": ("H2/H4 replaced, flipped or scaled slices (row/column/channel norm ratio outside [1/1.6, 1.6], or cosine < 0.3 "
               "against the parent): rescale a pure scale, transplant the parent's slice otherwise.",
               "CER falls by 0.3-4 pp; if refused, the slice drift is genuine training and not the defect"),
    "spikes": ("H3 spikes: |w| > 8 rms of the tensor and moved > 4 rms from the parent (suspect L_16 forward ih, abs max 4.70); "
               "set them back to the parent's values.",
               "CER falls by 0.5-2.5 pp, mostly through collapsed lines 10 -> <= 3; if refused, the spikes are harmless and the "
               "collapses have another cause"),
    "blocks": ("H4b gate / direction / whole-tensor blocks with norm ratio outside [1/1.25, 1.25] against the parent: rescale back "
               "(cos > 0.9) or transplant.",
               "CER falls by 0.3-2 pp; if refused, natural drift"),
    "bias": ("H5 bias outliers: O_18 row bias drift vs parent more than 3.0 from the median (2.0 from the frequent-row median "
             "for frequent letters, punctuation, blank, space); LSTM/conv bias entries more than max(0.3, 8 MAD) from the "
             "median drift: recentre to parent + median drift.",
             "CER falls by 0.3-2 pp for a frequent row, 0.5-0.8 pp if the space row is the outlier (103 space deletions); "
             "if refused, the shift is the model's own calibration"),
}


def run_edit(label: str, text: str, prediction: str, edit_class: str, deltas: dict) -> bool | None:
    """hypothesis -> one add over the named tensors -> measure -> own collapse check -> s.keep. Returns kept or None."""
    if s.budget_left < 1:
        s.note(f"{label}: no pass left, edit not applied")
        return None
    h = s.hypothesis(text, prediction, edit_class)
    names = sorted(deltas)
    per = {n: {"delta": np.asarray(deltas[n], dtype=np.float32)} for n in names}
    s.edit("add", names, hypothesis=h, per_tensor=per)
    m = s.measure(label)
    d = m.cer - state["cer"]
    s.note(f"{label}: CER {m.cer:.4f} ({d:+.4f} vs kept {state['cer']:.4f}), WER {m.wer:.4f}, collapsed {m.collapsed} "
           f"(kept {state['collapsed']}); top confusions: {top_conf(m)}")
    if m.collapsed > state["collapsed"] + MAX_COLLAPSE_RISE:
        s.revert(f"{label}: collapsed lines rose {state['collapsed']} -> {m.collapsed}; not a repair whatever the CER")
        return False
    ok = s.keep(f"{label}: CER {d:+.4f} against the kept state; the rule decides")
    if ok:
        state.update(cer=m.cer, wer=m.wer, collapsed=m.collapsed)
        kept_log.append(f"{label} (CER {d:+.4f}, collapsed {m.collapsed})")
    s.note(f"{label}: {'kept' if ok else 'refused by the rule, weights restored'}")
    return ok


# ---------------------------------------------------------------- the session
try:
    # pass 1: baseline
    m0 = s.measure("C0")
    state.update(cer=m0.cer, wer=m0.wer, collapsed=m0.collapsed)
    s.note(f"C0: CER {m0.cer:.4f} WER {m0.wer:.4f} collapsed {m0.collapsed} (pack: 0.0726 / 0.1923 / 10); "
           f"top confusions: {top_conf(m0)}")

    # structural signatures, one edit per flagged class, recomputed from the current weights before each edit
    structural = 0
    space_fixed = False
    sp_row = cm.get(" ")
    for cls in ("zeros", "slices", "spikes", "blocks", "bias"):
        if structural >= MAX_STRUCTURAL or s.budget_left < 1:
            s.note(f"{cls}: skipped (structural edits {structural}, passes left {s.budget_left})")
            continue
        deltas, descs = COMPUTE[cls]()
        if not deltas:
            s.note(f"{cls}: nothing flagged against the parent; no edit, no pass")
            continue
        total = sum(float(np.linalg.norm(d)) for d in deltas.values())
        text, pred = HYP[cls]
        text = f"{text} Flagged ({len(descs)}, total delta L2 {total:.3f}): " + "; ".join(descs)[:2500]
        ok = run_edit(cls, text, pred, "add (masked restore / rescale from the parent)", deltas)
        structural += 1
        if cls == "bias" and ok and sp_row is not None and "O_18.lin.bias" in deltas and deltas["O_18.lin.bias"][sp_row] != 0:
            space_fixed = True

    # calibration probes, independent of a plant
    nb = T["O_18.lin.bias"].shape[0]
    if sp_row is not None and not space_fixed:
        d = np.zeros(nb, np.float32)
        d[sp_row] = 0.75
        run_edit("P1 space +0.75",
                 "P1 space bias +0.75: 103 space deletions against 22 insertions at C0 say the space row is under-biased.",
                 "CER falls by 0.15-0.35 pp (space deletions fall more than insertions rise); if refused, the asymmetry is the "
                 "hand / the collapsed lines, not a bias defect",
                 "add (one bias row)", {"O_18.lin.bias": d})
    else:
        s.note("P1 skipped: space row already repaired under H5 and kept" if space_fixed else "P1 skipped: no space row")

    d = np.zeros(nb, np.float32)
    d[0] = -0.4
    run_edit("P2 blank -0.4",
             "P2 blank bias -0.4: the reading is deletion-heavy (length ratio 0.971, 10 collapsed lines); a slightly over-firing "
             "blank would explain it.",
             "CER falls by 0.1-0.3 pp if the blank over-fires; if refused, the blank is not over-firing and the deletions sit in "
             "the collapsed lines",
             "add (blank bias)", {"O_18.lin.bias": d})

    new_rows = [cm[c] for c in cm if c not in pm and c != "⸗" and cm[c] < nb]
    if new_rows:
        d = np.zeros(nb, np.float32)
        d[new_rows] = -1.5
        run_edit("P3 new rows -1.5",
                 f"P3 child-only output rows ({len(new_rows)} rows, not ⸗) have biases around -0.65, far above the parent's rare "
                 "rows (-2 to -7); lower them by 1.5 so they stop firing spuriously on 19th-century German.",
                 "CER falls by 0.05-0.3 pp through fewer spurious rare-character outputs; 0 if they never fire; if refused, "
                 "harmless and no gain",
                 "add (bias rows)", {"O_18.lin.bias": d})
    else:
        s.note("P3 skipped: no child-only rows")

    s.note("Conclusion: " + (f"kept {len(kept_log)} edit(s): " + "; ".join(kept_log) +
                             f"; CER {state['cer']:.4f} collapsed {state['collapsed']} (C0 {m0.cer:.4f} / {m0.collapsed})."
                             if kept_log else
                             f"nothing kept: no plant signature in the parent diff and no calibration gain; the model stands "
                             f"at CER {m0.cer:.4f} with {m0.collapsed} collapsed lines - no defect found at the resolution of these readings.")
           + f" Passes used {s.budget - s.budget_left} of {s.budget}.")
except Exception as e:                                        # never leave the session without a freeze
    s.note(f"program error: {e!r}\n{traceback.format_exc()[-1500:]}")
    if getattr(s, "_last_edit", None) is not None:
        s.revert("program error after the edit")
finally:
    summary = s.freeze()
    print(summary.get("used"), "passes used;", summary.get("kept"), "kept;", summary.get("final_digest"))
