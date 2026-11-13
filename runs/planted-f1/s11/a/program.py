"""planted-f1 / s11 / task (a): programmer with 10 passes. Written in full before the first pass.

Readings (pack): the four L_14 weight matrices sit 0.75 / 0.55 rel L2 from the parent (cos 0.79 / 0.87) while L_12 and
L_16 sit at 0.30 / 0.21, L_14's biases moved normally and its norms grew 13-21 %: the excess fits i.i.d. noise of one
std (~0.15) on every entry. The space row carries the most negative output bias of all classes (-7.85) and ' ' -> nothing
is the top confusion (156). L_16.weight_ih_l0 has an abs max of 4.7 (other LSTM weights 1.4-2.4).
Hypotheses and predictions: preregistration_hypotheses.md (H1-H6). Only wsp + numpy; the parent is the donor.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
NOTHING = ("", None, "∅")          # how a deletion/insertion partner may be written in m.confusions

s = open_session(".")
T = s.subject.tensors()                 # live view
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
P = parent.tensors()
pm = parent.codec_map()

kept_m = None                           # Metrics of the last kept state (own checks)


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def conf_items(m):
    """[(ref, hyp, count)] from m.confusions whatever its container shape; [] if unreadable."""
    out = []
    try:
        conf = m.confusions
        if isinstance(conf, dict):
            for k, v in conf.items():
                if isinstance(k, (tuple, list)) and len(k) == 2:
                    out.append((k[0], k[1], int(v)))
                elif isinstance(v, dict):                       # nested {ref: {hyp: count}}
                    for h, c in v.items():
                        out.append((k, h, int(c)))
        else:
            for it in conf:
                if isinstance(it, dict):
                    r = it.get("ref", it.get("reference", it.get("a")))
                    h = it.get("hyp", it.get("hypothesis", it.get("b")))
                    c = it.get("count", it.get("n", 1))
                    out.append((r, h, int(c)))
                elif len(it) == 3:
                    out.append((it[0], it[1], int(it[2])))
                elif len(it) == 2 and isinstance(it[0], (tuple, list)) and len(it[0]) == 2:
                    out.append((it[0][0], it[0][1], int(it[1])))
    except Exception as e:  # noqa: BLE001
        s.note(f"confusions unreadable: {e!r}")
        return []
    return out


def del_ins(m):
    """(deletions, insertions) summed over the reported confusions, or None."""
    items = conf_items(m)
    if not items:
        return None
    dels = sum(c for r, h, c in items if h in NOTHING and r not in NOTHING)
    ins = sum(c for r, h, c in items if r in NOTHING and h not in NOTHING)
    return dels, ins


def pair_count(m, ref, hyp_nothing=True):
    """count of ref -> nothing (or nothing -> ref) in the reported confusions; None if unreadable."""
    items = conf_items(m)
    if not items:
        return None
    if hyp_nothing:
        return sum(c for r, h, c in items if r == ref and h in NOTHING)
    return sum(c for r, h, c in items if h == ref and r in NOTHING)


def fmt(m):
    return f"cer {m.cer:.4f} wer {m.wer:.4f} collapsed {m.collapsed}"


def pending():
    return getattr(s, "_last_edit", None) is not None


def attempt(text, prediction, edit_class, op, names, label, reason, per_tensor=None, **kwargs):
    """hypothesis -> edit -> one pass -> own check -> s.keep (the harness rule decides). Returns (kept, Metrics|None)."""
    global kept_m
    if s.budget_left < 1:
        s.note(f"skip {label}: no pass left")
        return False, None
    h = s.hypothesis(text, prediction, edit_class)
    snap = {n: np.array(T[n], copy=True) for n in names}
    try:
        s.edit(op, list(names), hypothesis=h, per_tensor=per_tensor, **kwargs)
    except Exception as e:  # noqa: BLE001
        s.note(f"{label}: edit failed before any pass ({e!r}); tensors restored")
        for n, v in snap.items():
            s.subject.set_tensor(n, v)
        if pending():
            s.revert(f"edit failed: {e!r}")
        return False, None
    try:
        m = s.measure(label)
    except Exception as e:  # noqa: BLE001
        s.revert(f"measure failed: {e!r}")
        return False, None
    print(label, fmt(m))
    if kept_m is not None and m.collapsed > kept_m.collapsed + 5:
        s.revert(f"own check: collapsed lines rose {kept_m.collapsed} -> {m.collapsed}")
        return False, m
    ok = s.keep(reason)
    s.note(f"{label}: {'KEPT' if ok else 'refused'} ({fmt(m)})")
    if ok:
        kept_m = m
    return ok, m


def row_norms(a):
    return np.linalg.norm(a.reshape(a.shape[0], -1).astype(np.float64), axis=1)


try:
    # ------------------------------------------------------------------------------------------------------------
    # pass 1: baseline
    # ------------------------------------------------------------------------------------------------------------
    m0 = s.measure("C0")
    kept_m = m0
    print("C0", fmt(m0))
    di0 = del_ins(m0)
    s.note(f"C0 {fmt(m0)}; deletions/insertions {di0}; space->nothing {pair_count(m0, ' ')}")

    # ------------------------------------------------------------------------------------------------------------
    # H1: noise on the four L_14 weight matrices -> Wiener interpolation toward the parent
    # ------------------------------------------------------------------------------------------------------------
    L14 = ["L_14.layer.weight_ih_l0", "L_14.layer.weight_hh_l0",
           "L_14.layer.weight_ih_l0_reverse", "L_14.layer.weight_hh_l0_reverse"]
    h1_kept = False
    try:
        est = []
        for n in L14:
            ref = n.replace("L_14", "L_16")
            d14 = T[n].astype(np.float64) - P[n].astype(np.float64)
            d16 = T[ref].astype(np.float64) - P[ref].astype(np.float64)
            v14, v16 = float(d14.var()), float(d16.var())
            excess = max(v14 - v16, 0.0)
            est.append((n, v14, v16, excess / v14 if v14 > 0 else 0.0, float(np.sqrt(excess))))
            s.note(f"{n}: var(child-parent) {v14:.5f} vs L_16 {v16:.5f}; noise std est {np.sqrt(excess):.4f}; "
                   f"rel L2 {np.linalg.norm(d14) / np.linalg.norm(P[n]):.3f}; alpha est {excess / v14 if v14 > 0 else 0:.3f}")
        ratios = [v14 / v16 if v16 > 0 else 9.9 for _, v14, v16, _, _ in est]
        if min(ratios) < 1.5:
            alpha = 0.6
            s.note(f"L_14 excess variance weak (ratios {ratios}); trying alpha 0.6 regardless")
        else:
            alpha = float(np.clip(np.mean([a for _, _, _, a, _ in est]), 0.6, 0.95))
        h1_kept, m1 = attempt(
            "H1: the four L_14 weight matrices carry i.i.d. additive noise (std ~0.15; rel L2 0.75/0.55 vs 0.30/0.21 "
            "for L_12/L_16, cos 0.79/0.87, norms +13-21 %, biases normal). Wiener interpolation toward the parent "
            f"with alpha {alpha:.3f} = excess variance / total variance removes most of it while keeping the "
            "child's adaptation share.",
            "CER falls by 0.03-0.06 (0.120 -> 0.06-0.09); collapsed lines 11 -> <= 5; deletions of e/n/r/i fall. "
            "If CER rises the displacement is the child's own training.",
            "block interpolation to parent (L_14 weights, biases untouched)",
            "interpolate", L14, f"H1 L_14 interpolate a={alpha:.2f}",
            "L_14 noise removed by interpolation toward the parent",
            per_tensor={n: {"other": P[n]} for n in L14}, alpha=alpha)
        if not h1_kept and s.budget_left >= 1:
            h1_kept, m1b = attempt(
                "H1 fallback: milder interpolation (alpha 0.4) keeps 60 % of the child's L_14 adaptation while "
                "removing 40 % of a noise term; distinguishes 'parent incompatible' from 'no noise'.",
                "CER falls by 0.01-0.03 if the noise reading is right; otherwise no change or a rise, and L_14 stays.",
                "block interpolation to parent (L_14 weights, alpha 0.4)",
                "interpolate", L14, "H1b L_14 interpolate a=0.40",
                "L_14 milder interpolation toward the parent",
                per_tensor={n: {"other": P[n]} for n in L14}, alpha=0.4)
            if not h1_kept:
                s.note("H1 concluded: L_14 displacement not repairable from the parent; L_14 left as is")
    except Exception as e:  # noqa: BLE001
        s.note(f"H1 block error: {e!r}")
        if pending():
            s.revert(f"H1 block error: {e!r}")

    # ------------------------------------------------------------------------------------------------------------
    # H2: output rows displaced far beyond the fine-tuning drift (space first)
    # ------------------------------------------------------------------------------------------------------------
    space_planted = False
    space_row = cm.get(" ")
    try:
        W, B = T["O_18.lin.weight"], T["O_18.lin.bias"]
        PW, PB = P["O_18.lin.weight"], P["O_18.lin.bias"]
        flagged_bias, flagged_w = [], []
        shared = [(ch, r) for ch, r in cm.items() if ch in pm]
        for ch, r in shared:
            pr = pm[ch]
            bdelta = float(B[r] - PB[pr])
            wrel = float(np.linalg.norm(W[r].astype(np.float64) - PW[pr]) / max(np.linalg.norm(PW[pr]), 1e-8))
            if abs(bdelta) > 3.5:
                flagged_bias.append((ch, r, pr, bdelta))
            if wrel > 0.9:
                flagged_w.append((ch, r, pr, wrel))
        if space_row is not None and " " in pm:
            sp_delta = float(B[space_row] - PB[pm[" "]])
            s.note(f"space: child bias {B[space_row]:+.3f}, parent {PB[pm[' ']]:+.3f} (delta {sp_delta:+.3f}); "
                   f"row norm child {np.linalg.norm(W[space_row]):.2f} parent {np.linalg.norm(PW[pm[' ']]):.2f}")
            if sp_delta < -2.0 and all(r != space_row for _, r, _, _ in flagged_bias):
                flagged_bias.append((" ", space_row, pm[" "], sp_delta))
        else:
            s.note("no single-label space row in one of the codecs; H2 space check skipped")
        s.note(f"H2 flagged bias rows: {[(c, round(d, 2)) for c, _, _, d in flagged_bias]}; "
               f"flagged weight rows: {[(c, round(w, 2)) for c, _, _, w in flagged_w]}")
        rows_b = sorted({r for _, r, _, _ in flagged_bias})
        rows_w = sorted({r for _, r, _, _ in flagged_w})
        n_rows = len(set(rows_b) | set(rows_w))
        if 1 <= n_rows <= 10:
            names, per = [], {}
            if rows_b:
                names.append("O_18.lin.bias")
                per["O_18.lin.bias"] = {"index": rows_b,
                                        "values": np.array([PB[pr] for _, r, pr, _ in sorted(flagged_bias, key=lambda t: t[1])
                                                            if r in rows_b], dtype=np.float32)}
            if rows_w:
                names.append("O_18.lin.weight")
                per["O_18.lin.weight"] = {"index": rows_w,
                                          "values": np.stack([PW[pr] for _, r, pr, _ in sorted(flagged_w, key=lambda t: t[1])
                                                              if r in rows_w]).astype(np.float32)}
            space_planted = space_row in rows_b
            chars = sorted({c for c, _, _, _ in flagged_bias} | {c for c, _, _, _ in flagged_w})
            kept2, m2 = attempt(
                f"H2: output rows {chars!r} sit far beyond the fine-tuning drift of the other 225 shared rows "
                "(bias delta beyond +-3.5 / weight rel L2 > 0.9 / space bias delta < -2 where every other frequent "
                "class sits near 0). Planted row damage; the parent's row (aligned by character) is the donor.",
                "CER falls by 0.005-0.012 (space -> nothing is 156 of 1544 edits); the space deletion count drops "
                "by at least half. If not, the deletions come from upstream.",
                "output row transplant from parent (bias, weights where flagged)",
                "set_rows", names, "H2 output rows from parent",
                "displaced output rows restored from the parent", per_tensor=per)
            if not kept2:
                space_planted = False
        elif n_rows > 10:
            s.note(f"H2: {n_rows} rows flagged, too many for a planted-row reading; nothing done")
        else:
            s.note("H2: no output row displaced beyond the thresholds; nothing planted on the output layer")
    except Exception as e:  # noqa: BLE001
        s.note(f"H2 block error: {e!r}")
        if pending():
            s.revert(f"H2 block error: {e!r}")

    # ------------------------------------------------------------------------------------------------------------
    # H4: conv pruning / channel damage (exact zeros, or whole output channels far from the parent)
    # ------------------------------------------------------------------------------------------------------------
    try:
        names, per, report = [], {}, []
        for c in ("C_0", "C_3", "C_6", "C_9"):
            wn, bn = f"{c}.co.weight", f"{c}.co.bias"
            Wc, PWc, Bc, PBc = T[wn], P[wn], T[bn], P[bn]
            zero_mask = (Wc == 0) & (PWc != 0)
            eq_frac = float((Wc == PWc).mean())
            dn = row_norms(Wc - PWc)
            pn = np.maximum(row_norms(PWc), 1e-8)
            rel = dn / pn
            ratio = row_norms(Wc) / pn
            bad = np.where((rel > 0.9) | (ratio < 0.5) | (ratio > 2.0))[0]
            report.append(f"{c}: zeros-where-parent-nonzero {zero_mask.mean():.3f}, equal-to-parent {eq_frac:.3f}, "
                          f"channel rel L2 max {rel.max():.2f}, bad channels {bad.tolist()}")
            delta = np.zeros_like(Wc)
            if zero_mask.mean() >= 0.05:
                delta[zero_mask] = PWc[zero_mask] - Wc[zero_mask]
            if 1 <= len(bad) <= 8:
                delta[bad] = PWc[bad] - Wc[bad]
                bdelta = np.zeros_like(Bc)
                bdelta[bad] = PBc[bad] - Bc[bad]
                names.append(bn)
                per[bn] = {"delta": bdelta}
            if np.any(delta != 0):
                names.append(wn)
                per[wn] = {"delta": delta}
        s.note("H4 scan: " + "; ".join(report))
        if names:
            attempt(
                f"H4: conv tensors {names!r} carry exact zeros where the parent is non-zero (>= 5 %) or whole output "
                "channels far from the parent (rel L2 > 0.9 or norm ratio outside [0.5, 2]); planted pruning/channel "
                "damage. Those entries/channels are reset to the parent's values.",
                "CER falls by 0.005-0.03; if it does not, the pattern is benign and stays.",
                "sparse/channel reset to parent (conv)",
                "add", names, "H4 conv reset to parent", "conv zeros/channels reset to the parent", per_tensor=per)
        else:
            s.note("H4: no conv zero/channel anomaly; no pass spent")
    except Exception as e:  # noqa: BLE001
        s.note(f"H4 block error: {e!r}")
        if pending():
            s.revert(f"H4 block error: {e!r}")

    # ------------------------------------------------------------------------------------------------------------
    # H3: localized damage in L_12 / L_16 weights (spikes, or gate rows far from the parent)
    # ------------------------------------------------------------------------------------------------------------
    try:
        names, per, report, n_spk, n_rows = [], {}, [], 0, 0
        for blk in ("L_12", "L_16"):
            for kind in ("weight_ih_l0", "weight_hh_l0", "weight_ih_l0_reverse", "weight_hh_l0_reverse"):
                n = f"{blk}.layer.{kind}"
                Wl, PWl = T[n], P[n]
                spk = (np.abs(Wl) > 3.0) & (np.abs(Wl - PWl) > 1.5)
                pn = np.maximum(row_norms(PWl), 1e-8)
                rel = row_norms(Wl - PWl) / pn
                ratio = row_norms(Wl) / pn
                bad = np.where((rel > 1.2) | (ratio < 0.5) | (ratio > 2.0))[0]
                report.append(f"{n}: abs max {np.abs(Wl).max():.2f}, spikes {int(spk.sum())}, "
                              f"row rel max {rel.max():.2f}, bad rows {len(bad)}")
                delta = np.zeros_like(Wl)
                if 1 <= int(spk.sum()) <= 400:
                    delta[spk] = PWl[spk] - Wl[spk]
                    n_spk += int(spk.sum())
                if 1 <= len(bad) <= 40:
                    delta[bad] = PWl[bad] - Wl[bad]
                    n_rows += len(bad)
                if np.any(delta != 0):
                    names.append(n)
                    per[n] = {"delta": delta}
        s.note("H3 scan: " + "; ".join(report))
        if names:
            attempt(
                f"H3: {n_spk} spike entries (|w| > 3, |w - parent| > 1.5) and {n_rows} gate rows far from the parent in "
                f"{names!r}; localized planted damage. Reset to the parent's values.",
                "CER falls by 0.001-0.01; if not, the outliers are natural and stay.",
                "sparse reset to parent (LSTM weights)",
                "add", names, "H3 LSTM spikes/rows reset", "LSTM spikes/rows reset to the parent", per_tensor=per)
        else:
            s.note("H3: no spike/row anomaly in L_12/L_16; no pass spent")
    except Exception as e:  # noqa: BLE001
        s.note(f"H3 block error: {e!r}")
        if pending():
            s.revert(f"H3 block error: {e!r}")

    # ------------------------------------------------------------------------------------------------------------
    # H5: CTC calibration — blank bias down if deletions still dominate
    # ------------------------------------------------------------------------------------------------------------
    try:
        di = del_ins(kept_m)
        s.note(f"after repairs: {fmt(kept_m)}; deletions/insertions {di}")
        if di is None or di[0] > 1.5 * di[1]:
            nb = T["O_18.lin.bias"].shape[0]
            d = np.zeros(nb, dtype=np.float32)
            d[0] = -0.5
            k5, _ = attempt(
                f"H5: deletions outnumber insertions ({di}); the blank is over-emitted. Blank bias -0.5 (calibration, "
                "not a defect repair).",
                "CER falls by 0.002-0.008; deletions fall, insertions rise less. If refused the deletion profile is the "
                "model's operating point.",
                "scalar bias shift (blank)", "add", ["O_18.lin.bias"], "H5 blank bias -0.5",
                "blank bias lowered 0.5", delta=d)
            if k5 and s.budget_left >= 1:
                di = del_ins(kept_m)
                if di is None or di[0] > 1.5 * di[1]:
                    attempt(
                        f"H5b: still deletion-dominated ({di}); a second -0.5 on the blank bias.",
                        "CER falls by 0.001-0.005; if refused, -0.5 was the right amount.",
                        "scalar bias shift (blank)", "add", ["O_18.lin.bias"], "H5b blank bias -0.5 more",
                        "blank bias lowered another 0.5", delta=d)
        else:
            s.note("H5: deletions no longer dominate; no calibration")
    except Exception as e:  # noqa: BLE001
        s.note(f"H5 block error: {e!r}")
        if pending():
            s.revert(f"H5 block error: {e!r}")

    # ------------------------------------------------------------------------------------------------------------
    # H6: space calibration, only if nothing planted was found on the space row
    # ------------------------------------------------------------------------------------------------------------
    try:
        if space_row is not None and not space_planted and s.budget_left >= 1:
            sp_del = pair_count(kept_m, " ")
            sp_ins = pair_count(kept_m, " ", hyp_nothing=False)
            s.note(f"H6: space -> nothing {sp_del}, nothing -> space {sp_ins}")
            if sp_del is None or sp_ins is None or sp_del > 3 * max(sp_ins, 1):
                nb = T["O_18.lin.bias"].shape[0]
                d = np.zeros(nb, dtype=np.float32)
                d[space_row] = 2.0
                attempt(
                    f"H6: the space bias (-7.85, the most negative of all classes) is natural per the parent but space "
                    f"deletions dominate ({sp_del} vs {sp_ins}); +2.0 on the space bias as calibration.",
                    "CER falls by 0.002-0.006; if refused the space suppression is upstream and the bias stays.",
                    "scalar bias shift (space)", "add", ["O_18.lin.bias"], "H6 space bias +2.0",
                    "space bias raised 2.0", delta=d)
            else:
                s.note("H6: space deletions not dominant; nothing done")
    except Exception as e:  # noqa: BLE001
        s.note(f"H6 block error: {e!r}")
        if pending():
            s.revert(f"H6 block error: {e!r}")

finally:
    if pending():
        s.revert("program ended with an undecided edit; reverted")
    summary = s.freeze()
    print({k: summary.get(k) for k in ("used", "edits", "kept", "baseline", "last_pass")})
