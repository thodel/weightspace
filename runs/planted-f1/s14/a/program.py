"""s14 task (a), budget 10 — the programmer's script (claude-fable-5-1), written in full before the first pass.

Pass 1 is the C0 baseline. Then diagnostics against the parent (no passes) and the stages of
preregistration_hypotheses.md in order: D1 output rows swapped/destroyed, D2 rows and D3 columns of LSTM/conv
weights rescaled or zeroed, D4 point spikes, D5 whole-tensor scale, D6 output bias shifts (space), D7 blank row norm,
P1 space bias probe, P2 blank bias probe, P3 C_3.co.bias transplant, P4/P5 second steps. Every edit: s.hypothesis ->
s.edit -> one s.measure -> s.keep (harness rule) or s.revert (own veto: collapsed lines up by > 3). At most 10 measures.
Runs inside the sandbox: wsp + numpy only.
"""
from __future__ import annotations

import traceback
from collections import Counter

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"
MAX_MEASURES = 10
NOTHING = (None, "", "∅")
# from pack/c0_practice.md (same practice set); used only if m.confusions cannot be parsed
PACK_SPACE_DEL, PACK_DELS, PACK_INS = 78, 230, 25

s = open_session(".")
T = s.subject.tensors()                 # live view: re-read T[name] after every edit
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
P = parent.tensors()
pm = parent.codec_map()
rmap = codec_row_map(cm, pm)            # child row -> parent row, shared characters
common = [n for n in T if n in P and T[n].shape == P[n].shape]
freq = Counter("".join(it.get("text", "") for it in s.practice))

state = {"kept_cer": None, "kept_collapsed": None, "measures": 0}


# ----------------------------------------------------------------------------------------------- helpers
def note(txt: str) -> None:
    print(txt, flush=True)
    try:
        s.note(txt)
    except Exception:
        pass


def passes_left() -> int:
    return min(s.budget_left, MAX_MEASURES - state["measures"])


def measure(label: str):
    m = s.measure(label)
    state["measures"] += 1
    return m


def conf_items(m) -> list:
    """(ref, hyp, count) triples from Metrics.confusions, whatever its exact container; [] if unknown."""
    out = []
    c = getattr(m, "confusions", None)
    try:
        if c is None:
            return out
        if isinstance(c, dict):
            for k, v in c.items():
                if isinstance(k, (tuple, list)) and len(k) == 2:
                    out.append((k[0], k[1], int(v)))
                elif isinstance(v, dict):
                    for h, n in v.items():
                        out.append((k, h, int(n)))
        else:
            for it in c:
                if isinstance(it, dict):
                    r = it.get("ref", it.get("reference"))
                    h = it.get("hyp", it.get("hypothesis"))
                    n = it.get("count", it.get("n", 1))
                    out.append((r, h, int(n)))
                elif isinstance(it, (tuple, list)) and len(it) >= 2:
                    if isinstance(it[0], (tuple, list)) and len(it[0]) == 2:
                        out.append((it[0][0], it[0][1], int(it[1])))
                    elif len(it) >= 3 and isinstance(it[2], (int, float, np.integer, np.floating)):
                        out.append((it[0], it[1], int(it[2])))
                    else:
                        out.append((it[0], it[1], 1))
    except Exception as e:
        note(f"confusions not parsed: {e!r}")
        return []
    return out


def error_profile(m):
    """(deletions, insertions, space deletions, parsed?)"""
    items = conf_items(m)
    if not items:
        return PACK_DELS, PACK_INS, PACK_SPACE_DEL, False
    dels = sum(n for r, h, n in items if h in NOTHING and r not in NOTHING)
    ins = sum(n for r, h, n in items if r in NOTHING and h not in NOTHING)
    sdel = sum(n for r, h, n in items if r == " " and h in NOTHING)
    return dels, ins, sdel, True


def trial(text, prediction, edit_class, op, names, label, reason, **kwargs):
    """hypothesis -> edit -> one measure -> keep (rule) / revert (own veto). Returns (kept, metrics)."""
    if passes_left() < 1:
        note(f"skip, no pass left: {text}")
        return False, None
    names = list(names)
    before = {n: T[n].copy() for n in names}
    h = s.hypothesis(text, prediction, edit_class)
    try:
        s.edit(op, names, hypothesis=h, **kwargs)
    except Exception as e:
        note(f"edit {op} on {names} failed: {e!r}; tensors restored, no pass spent")
        for n, v in before.items():
            s.subject.set_tensor(n, v)
        if getattr(s, "_last_edit", None) is not None:
            s.revert("edit raised")
        return False, None
    m = measure(label)
    kc, kk = state["kept_cer"], state["kept_collapsed"]
    note(f"[{label}] cer {m.cer:.4f} (kept {kc:.4f}, delta {m.cer - kc:+.4f}) wer {m.wer:.4f} "
         f"collapsed {m.collapsed} (kept {kk})")
    if m.collapsed > kk + 3:
        s.revert(f"{reason} -- own veto: collapsed lines rose {kk} -> {m.collapsed}")
        note("reverted by own veto (collapsed lines rose)")
        return False, m
    kept = s.keep(reason)
    note("kept" if kept else "refused by the keep rule; undone")
    if kept:
        state["kept_cer"], state["kept_collapsed"] = m.cer, m.collapsed
    return kept, m


def axis_norms(w: np.ndarray, axis: int) -> np.ndarray:
    w2 = np.moveaxis(w, axis, 0).reshape(w.shape[axis], -1)
    return np.linalg.norm(w2, axis=1)


def run_stage(fn, *args):
    """A stage must never leave an undecided edit or kill the program."""
    try:
        return fn(*args)
    except Exception:
        note(f"stage {fn.__name__} raised:\n{traceback.format_exc()}")
        if getattr(s, "_last_edit", None) is not None:
            s.revert(f"stage {fn.__name__} raised")
        return None


# ----------------------------------------------------------------------------------------------- D1: output rows
def shared_rows():
    chars = [ch for ch in cm if ch in pm]
    crow = np.array([cm[ch] for ch in chars], dtype=int)
    prow = np.array([pm[ch] for ch in chars], dtype=int)
    return chars, crow, prow


def stage_output_rows():
    chars, crow, prow = shared_rows()
    cw, cb = T[W].astype(np.float64), T[B].astype(np.float64)
    pw, pb = P[W].astype(np.float64), P[B].astype(np.float64)
    C, Pp = cw[crow], pw[prow]
    cn, pn = np.linalg.norm(C, axis=1), np.linalg.norm(Pp, axis=1)
    Cn = C / np.maximum(cn, 1e-9)[:, None]
    Pn = Pp / np.maximum(pn, 1e-9)[:, None]
    M = Cn @ Pn.T                                   # child i vs parent j
    own = np.diag(M).copy()
    ratio = cn / np.maximum(pn, 1e-9)
    dbias = cb[crow] - pb[prow]
    fi = [i for i, ch in enumerate(chars) if freq.get(ch, 0) >= 30]
    if len(fi) < 5:
        note("D1: fewer than 5 frequent shared characters; stage skipped")
        return
    med_r, med_b = float(np.median(ratio[fi])), float(np.median(dbias[fi]))
    note(f"D1 output rows: {len(chars)} shared, {len(fi)} frequent; median norm ratio {med_r:.3f}, "
         f"median bias delta {med_b:+.3f}; all-rows mean cos {own.mean():.3f}")
    order = sorted(fi, key=lambda i: -freq[chars[i]])
    note("D1 frequent rows (char count | norm child/parent ratio | cos | bias child parent): " + "; ".join(
        f"{chars[i]!r} {freq[chars[i]]} | {cn[i]:.1f}/{pn[i]:.1f} {ratio[i]:.2f} | {own[i]:.2f} | "
        f"{cb[crow[i]]:+.2f} {pb[prow[i]]:+.2f}" for i in order))
    fixes = {}
    swapped = set()
    for i in fi:
        j = int(np.argmax(M[i]))
        if j != i and i not in swapped and j not in swapped and M[i, j] > own[i] + 0.25 and M[j, i] > own[j] + 0.25:
            note(f"D1 swap suspected: child {chars[i]!r} matches parent {chars[j]!r} ({M[i, j]:.2f} vs own {own[i]:.2f}), "
                 f"child {chars[j]!r} matches parent {chars[i]!r} ({M[j, i]:.2f} vs own {own[j]:.2f})")
            fixes[int(crow[i])] = (cw[crow[j]].copy(), float(cb[crow[j]]))
            fixes[int(crow[j])] = (cw[crow[i]].copy(), float(cb[crow[i]]))
            swapped |= {i, j}
    for i in fi:
        if i in swapped:
            continue
        if own[i] < 0.2 or ratio[i] < 0.15 * med_r:
            note(f"D1 destroyed row suspected: {chars[i]!r} cos {own[i]:.2f}, ratio {ratio[i]:.2f}; rebuilt from the parent")
            fixes[int(crow[i])] = (pw[prow[i]] * med_r, float(pb[prow[i]] + med_b))
        elif ratio[i] < 0.3 * med_r or ratio[i] > 3.0 * med_r:
            note(f"D1 rescaled row suspected: {chars[i]!r} ratio {ratio[i]:.2f} vs median {med_r:.2f}; rescaled to median")
            fixes[int(crow[i])] = (cw[crow[i]] * (med_r * pn[i] / max(cn[i], 1e-9)), float(cb[crow[i]]))
    if not fixes:
        note("D1: no swapped, destroyed or rescaled frequent output row; no edit")
        return
    idx = sorted(fixes)
    vals_w = np.stack([fixes[r][0] for r in idx]).astype(np.float32)
    vals_b = np.array([fixes[r][1] for r in idx], dtype=np.float32)
    trial(f"D1: output rows of {[k for k, v in cm.items() if v in idx]} are swapped, destroyed or rescaled against the parent",
          "CER down 1 to 4 points on the practice set", "set_rows O_18 weight+bias",
          "set_rows", [W, B], "D1 output rows", "D1 output rows repaired from parent alignment",
          per_tensor={W: {"index": idx, "values": vals_w}, B: {"index": idx, "values": vals_b}})


# ----------------------------------------------------------------------------------------------- D2/D3: rows, columns
def stage_axis(axis: int, tag: str):
    op = "set_rows" if axis == 0 else "set_cols"
    per_tensor = {}
    report = []
    for n in common:
        if T[n].ndim < 2 or T[n].shape[axis] < 4:
            continue
        c, p = T[n].astype(np.float64), P[n].astype(np.float64)
        nc, npar = axis_norms(c, axis), axis_norms(p, axis)
        ok = npar > 1e-6
        if ok.sum() < 4:
            continue
        ratio = np.where(ok, nc / np.maximum(npar, 1e-9), 1.0)
        med = float(np.median(ratio[ok]))
        flag = ok & ((ratio < 0.3 * med) | (ratio > 2.5 * med))
        idx = np.nonzero(flag)[0]
        report.append(f"{n}: median {med:.2f}, range {ratio[ok].min():.2f}..{ratio[ok].max():.2f}, flagged {len(idx)}")
        if len(idx) == 0:
            continue
        if len(idx) > 0.25 * len(ratio):
            note(f"{tag} {n}: {len(idx)} of {len(ratio)} flagged -- systematic, left to D5 (scale)")
            continue
        vals = []
        for i in idx:
            ci, pi = np.take(c, i, axis=axis), np.take(p, i, axis=axis)
            vals.append(ci * (med * npar[i] / nc[i]) if nc[i] > 1e-6 else pi * med)
        per_tensor[n] = {"index": [int(i) for i in idx],
                         "values": np.stack(vals, axis=axis).astype(np.float32)}
        note(f"{tag} {n}: flagged {[int(i) for i in idx][:40]} ratios "
             f"{[round(float(ratio[i]), 2) for i in idx][:40]}")
    note(f"{tag} axis-{axis} norm ratios child/parent: " + " | ".join(report))
    if not per_tensor:
        note(f"{tag}: no rescaled or zeroed {'rows' if axis == 0 else 'columns'}; no edit")
        return
    trial(f"{tag}: {'rows' if axis == 0 else 'columns'} of {list(per_tensor)} are rescaled or zeroed against the parent",
          "CER down 0.5 to 3 points", f"{op} on LSTM/conv weights",
          op, list(per_tensor), f"{tag} axis {axis}", f"{tag} rows/columns restored to the median child/parent ratio",
          per_tensor=per_tensor)


# ----------------------------------------------------------------------------------------------- D4: spikes
def stage_spikes():
    per_tensor, report = {}, []
    for n in common:
        c, p = T[n].astype(np.float64), P[n].astype(np.float64)
        d = c - p
        srms = float(np.sqrt(np.mean(d * d))) + 1e-12
        pmax = float(np.abs(p).max()) + 1e-12
        mask = (np.abs(d) > 14 * srms) & (np.abs(d) > 0.5)
        mask |= (np.abs(c) > 1.6 * pmax) & (np.abs(d) > 0.5)
        k = int(mask.sum())
        report.append(f"{n}: absmax child {np.abs(c).max():.3f} parent {pmax:.3f}, diff rms {srms:.4f} "
                      f"max {np.abs(d).max():.3f}, flagged {k}")
        if k == 0:
            continue
        if k > 0.02 * c.size:
            note(f"D4 {n}: {k} flagged entries (> 2 %), not a point defect; skipped here")
            continue
        per_tensor[n] = {"delta": np.where(mask, -d, 0.0).astype(np.float32)}
        pos = np.argwhere(mask)
        note(f"D4 spikes in {n}: {k} entries, |child| up to {np.abs(c[mask]).max():.3f}, |diff| up to "
             f"{np.abs(d[mask]).max():.3f}; first positions {pos[:12].tolist()}")
    note("D4 spike scan: " + " | ".join(report))
    if not per_tensor:
        note("D4: no point spikes against the parent; no edit")
        return
    trial(f"D4: point spikes in {list(per_tensor)} (entries far outside the parent's range) are planted",
          "CER down 0.3 to 2 points, collapsed lines down", "add (revert flagged entries to parent)",
          "add", list(per_tensor), "D4 spikes", "D4 spike entries reverted to the parent's values",
          per_tensor=per_tensor)


# ----------------------------------------------------------------------------------------------- D5: tensor scale
def stage_scale():
    per_tensor, report = {}, []
    for n in common:
        c, p = T[n].astype(np.float64).ravel(), P[n].astype(np.float64).ravel()
        pp = float(p @ p)
        if pp < 1.0:
            continue
        k = float(c @ p / pp)
        cos = float(c @ p / (np.linalg.norm(c) * np.sqrt(pp) + 1e-12))
        report.append(f"{n}: k {k:.3f} cos {cos:.3f} norm ratio {np.linalg.norm(c) / np.sqrt(pp):.3f}")
        if abs(k - 1.0) > 0.2:
            per_tensor[n] = {"factor": float(1.0 / k)}
            note(f"D5 {n}: least-squares scale to parent k = {k:.3f}; rescale by {1.0 / k:.3f}")
    note("D5 scale scan: " + " | ".join(report))
    if not per_tensor:
        note("D5: no tensor scaled against the parent (|k-1| <= 0.2); no edit")
        return
    trial(f"D5: {list(per_tensor)} carry a planted whole-tensor scale", "CER down 0.5 to 3 points", "scale",
          "scale", list(per_tensor), "D5 scale", "D5 whole-tensor scale undone", per_tensor=per_tensor)


# ----------------------------------------------------------------------------------------------- D6: output bias
def stage_output_bias():
    chars, crow, prow = shared_rows()
    cb, pb = T[B].astype(np.float64), P[B].astype(np.float64)
    dbias = cb[crow] - pb[prow]
    fi = [i for i, ch in enumerate(chars) if freq.get(ch, 0) >= 30]
    if len(fi) < 5:
        note("D6: too few frequent shared characters; skipped")
        return set()
    med_b = float(np.median(dbias[fi]))
    flagged = [i for i in fi if abs(dbias[i] - med_b) > 2.5]
    sp = f"space bias child {cb[cm[' ']]:+.2f} parent {pb[pm[' ']]:+.2f}" if " " in cm and " " in pm else "no space row"
    note(f"D6 bias deltas of frequent rows: median {med_b:+.3f}, spread {dbias[fi].min():+.2f}..{dbias[fi].max():+.2f}; {sp}; "
         f"flagged {[chars[i] for i in flagged]}")
    if not flagged:
        note("D6: no frequent row with a shifted bias; no edit")
        return set()
    idx = [int(crow[i]) for i in flagged]
    vals = np.array([pb[prow[i]] + med_b for i in flagged], dtype=np.float32)
    kept, _ = trial(f"D6: output bias of {[chars[i] for i in flagged]} shifted by "
                    f"{[round(float(dbias[i] - med_b), 2) for i in flagged]} against parent+drift",
                    "CER down 0.3 to 0.8 points (space deletions 78 -> ~30 if the space is among them)",
                    "set_rows O_18.lin.bias", "set_rows", [B], "D6 output bias", "D6 bias set to parent + median drift",
                    index=idx, values=vals)
    return {chars[i] for i in flagged} if kept else set()


# ----------------------------------------------------------------------------------------------- D7: blank row
def stage_blank_row():
    c0, p0 = T[W][0].astype(np.float64), P[W][0].astype(np.float64)
    n_c, n_p = float(np.linalg.norm(c0)), float(np.linalg.norm(p0))
    cos = float(c0 @ p0 / (n_c * n_p + 1e-12))
    note(f"D7 blank row: norm child {n_c:.2f} parent {n_p:.2f} ratio {n_c / n_p:.3f}, cos {cos:.3f}, "
         f"bias child {T[B][0]:+.3f} parent {P[B][0]:+.3f}")
    r = n_c / n_p
    if 0.5 <= r <= 2.0:
        note("D7: blank row norm within [0.5, 2] of the parent; no edit")
        return
    trial(f"D7: the blank row is rescaled ({r:.2f} x the parent's norm)", "CER down 0.5 to 2 points, collapsed lines down",
          "set_rows O_18.lin.weight row 0 to parent norm", "set_rows", [W], "D7 blank row norm",
          "D7 blank row rescaled to the parent's norm", index=[0], values=(c0 * (n_p / n_c)).astype(np.float32)[None, :])


# ----------------------------------------------------------------------------------------------- probes
def bias_delta(row: int, amount: float) -> np.ndarray:
    d = np.zeros(T[B].shape, dtype=np.float32)
    d[row] = amount
    return d


def stage_probe_space(space_fixed: bool, sdel: int):
    if " " not in cm:
        note("P1: no single-label space row; skipped")
        return False
    if space_fixed:
        note("P1: space bias already repaired in D6; probe skipped")
        return False
    if sdel < 40:
        note(f"P1: only {sdel} space deletions at baseline; probe skipped")
        return False
    kept, _ = trial(f"P1: the space bias ({T[B][cm[' ']]:+.2f}) is too negative; +1.5",
                    "CER down 0.1 to 0.4 points; space deletions down, space insertions up", "add O_18.lin.bias",
                    "add", [B], "P1 space bias +1.5", "P1 space bias raised by 1.5", delta=bias_delta(cm[" "], 1.5))
    return kept


def stage_probe_blank(dels: int, ins: int, collapsed: int):
    if not (dels >= 2 * ins or collapsed >= 5):
        note(f"P2: deletions {dels} vs insertions {ins}, collapsed {collapsed}; blank probe skipped")
        return False
    kept, _ = trial(f"P2: the blank bias ({T[B][0]:+.2f}) is too high for this hand (deletions {dels} vs insertions {ins}, "
                    f"{collapsed} collapsed lines); -0.5",
                    "CER down 0.1 to 0.4 points; deletions and collapsed lines down, insertions up", "add O_18.lin.bias",
                    "add", [B], "P2 blank bias -0.5", "P2 blank bias lowered by 0.5", delta=bias_delta(0, -0.5))
    return kept


def stage_c3_bias():
    c, p = T["C_3.co.bias"].astype(np.float64), P["C_3.co.bias"].astype(np.float64)
    cos = float(c @ p / (np.linalg.norm(c) * np.linalg.norm(p) + 1e-12))
    note(f"P3 C_3.co.bias: norm child {np.linalg.norm(c):.4f} parent {np.linalg.norm(p):.4f}, cos {cos:.3f}")
    trial(f"P3: C_3.co.bias (cos {cos:.2f} to the parent, the only tensor below 0.9) was perturbed",
          "|CER change| < 0.1 points, most likely refused", "transplant from parent",
          "transplant", ["C_3.co.bias"], "P3 C_3 bias transplant", "P3 C_3.co.bias transplanted from the parent",
          donor=P["C_3.co.bias"].copy())


def stage_second_steps(blank_kept: bool, space_kept: bool):
    if blank_kept and passes_left() >= 1:
        trial("P4: a further -0.5 on the blank bias after the kept first step", "CER down < first step, or refused",
              "add O_18.lin.bias", "add", [B], "P4 blank bias -0.5 more", "P4 second blank step", delta=bias_delta(0, -0.5))
    if space_kept and " " in cm and passes_left() >= 1:
        trial("P5: a further +1.0 on the space bias after the kept first step", "CER down < first step, or refused",
              "add O_18.lin.bias", "add", [B], "P5 space bias +1.0 more", "P5 second space step",
              delta=bias_delta(cm[" "], 1.0))


# ----------------------------------------------------------------------------------------------- main
def main():
    note(f"start: budget {s.budget_left}, {len(s.practice)} practice lines, {len(common)} same-shape tensors, "
         f"{len(rmap)} shared output rows; frequent chars (>=30): {sum(1 for ch, n in freq.items() if n >= 30)}")
    m0 = measure("C0")
    state["kept_cer"], state["kept_collapsed"] = m0.cer, m0.collapsed
    dels, ins, sdel, parsed = error_profile(m0)
    note(f"[C0] cer {m0.cer:.4f} wer {m0.wer:.4f} collapsed {m0.collapsed}; deletions {dels} insertions {ins} "
         f"space deletions {sdel} ({'from confusions' if parsed else 'from the pack, confusions not parsed'})")

    run_stage(stage_output_rows)                       # D1
    run_stage(stage_axis, 0, "D2")                     # D2 rows
    run_stage(stage_axis, 1, "D3")                     # D3 columns
    run_stage(stage_spikes)                            # D4
    run_stage(stage_scale)                             # D5
    fixed = run_stage(stage_output_bias) or set()      # D6
    run_stage(stage_blank_row)                         # D7
    space_kept = bool(run_stage(stage_probe_space, " " in fixed, sdel))          # P1
    blank_kept = bool(run_stage(stage_probe_blank, dels, ins, m0.collapsed))     # P2
    run_stage(stage_c3_bias)                                                     # P3
    run_stage(stage_second_steps, blank_kept, space_kept)                        # P4/P5
    note(f"end: {state['measures']} measures, kept-state cer {state['kept_cer']:.4f} (baseline {m0.cer:.4f}), "
         f"collapsed {state['kept_collapsed']} (baseline {m0.collapsed})")


try:
    main()
except Exception:
    note(f"program error:\n{traceback.format_exc()}")
finally:
    if getattr(s, "_last_edit", None) is not None:
        s.revert("undecided edit at the end of the program")
    summary = s.freeze()
    print("frozen:", {k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})
