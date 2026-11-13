"""planted-f1 / s08 / task (a), budget 10 — written in full before the first pass.

Reading of the pack: output row `e` of O_18.lin.weight has norm 0.00 (zeroed); 910 e->∅ deletions and ~380 e->{a,i,o,r,c,u}
substitutions make up ~1290 of the 2088 baseline edits. A weak second anomaly sits in L_16.layer.weight_ih_l0 (abs max 4.70,
rms 7 % above its reverse twin). See preregistration_hypotheses.md for the hypotheses H0-H5 and the pass plan.
"""
from __future__ import annotations

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"   # the only donor (control only)
W_NAME, B_NAME = "O_18.lin.weight", "O_18.lin.bias"
FALLBACK_COMPONENTS = ["a", "i", "o", "r", "c", "u"]

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()


# ----------------------------------------------------------------------------- helpers
def row_norm(name: str, idx: int) -> float:
    return float(np.linalg.norm(T[name][idx]))


def conf_counts(m, ch: str) -> tuple[int, int, int]:
    """(deletions of ch, insertions/substitutions into ch, substitutions of ch into another char) from the top-10 confusions."""
    dele = ins = sub = 0
    for ref, hyp, n in m.confusions:
        if ref == ch and hyp == "":
            dele += n
        elif hyp == ch and ref != ch:
            ins += n
        elif ref == ch and hyp != "" and hyp != ch:
            sub += n
    return dele, ins, sub


def tried(h_text: str, prediction: str, edit_class: str, op: str, names: list[str], label: str, reason: str,
          rule_mask=None, **kwargs):
    """hypothesis -> edit -> one measure -> keep (the rule decides; refused edits are undone by the harness)."""
    if s.budget_left < 1:
        s.note(f"skipped (no pass left): {label}")
        return None, False
    h = s.hypothesis(h_text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, **kwargs)
    m = s.measure(label, rule_mask=rule_mask)
    kept = s.keep(reason)
    s.note(f"{label}: cer={m.cer:.4f} wer={m.wer:.4f} collapsed={m.collapsed} kept={kept} "
           f"top={m.confusions[:6]}")
    return m, kept


def set_e_row(h_text: str, prediction: str, edit_class: str, row: int, new_row: np.ndarray, new_bias: float,
              label: str, reason: str, rule_mask=None):
    return tried(h_text, prediction, edit_class, "set_rows", [W_NAME, B_NAME], label, reason, rule_mask=rule_mask,
                 index=[row],
                 per_tensor={W_NAME: {"values": new_row.astype(np.float32)},
                             B_NAME: {"values": np.array([new_bias], dtype=np.float32)}})


def consensus_row(components: list[str], weights: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """k * weighted mean of the component rows / biases, k so that the norm equals the median component norm."""
    W, b = T[W_NAME].astype(np.float64), T[B_NAME].astype(np.float64)
    idx = [cm[c] for c in components]
    w = weights / weights.sum()
    mean_row = (w[:, None] * W[idx]).sum(axis=0)
    mean_bias = float((w * b[idx]).sum())
    target = float(np.median(np.linalg.norm(W[idx], axis=1)))
    k = target / max(float(np.linalg.norm(mean_row)), 1e-9)
    return k * mean_row, k * mean_bias, k, target


# ----------------------------------------------------------------------------- pass 1: baseline (H0)
m0 = s.measure("C0")
s.note(f"C0: cer={m0.cer:.4f} wer={m0.wer:.4f} collapsed={m0.collapsed} top={m0.confusions}")

e_row = cm.get("e")
e_zero = e_row is not None and row_norm(W_NAME, e_row) < 1e-3
if e_row is None:
    s.note("no single-label row for 'e' in the codec; e-branch skipped")
else:
    s.note(f"row e = {e_row}, norm {row_norm(W_NAME, e_row):.4f}, bias {float(T[B_NAME][e_row]):+.3f}; zero={e_zero}")
    del0, ins0, sub0 = conf_counts(m0, "e")
    s.note(f"C0 e-errors in top-10: deletions {del0}, insertions {ins0}, substitutions {sub0}")

if e_zero:
    # ------------------------------------------------------------------------- pass 2: donor control (H1)
    try:
        donor = KrakenSubject(PARENT)
        pm = donor.codec_map()
        D = donor.tensors()
    except Exception as exc:  # the control is optional; the repair does not depend on the donor
        donor, pm, D = None, {}, {}
        s.note(f"donor could not be loaded: {exc!r}")
    if pm.get("e") is not None and D.get(W_NAME) is not None and D[W_NAME].shape[1] == T[W_NAME].shape[1]:
        d_row = D[W_NAME][pm["e"]].astype(np.float64)
        d_bias = float(D[B_NAME][pm["e"]])
        mask = s.unseen_by("kraken-bern-hertwig-v2")
        rule_mask = mask if sum(mask) >= 100 else None
        s.note(f"donor e row: norm {np.linalg.norm(d_row):.3f}, bias {d_bias:+.3f}; rule_mask lines={sum(mask)}")
        set_e_row("H1 control: the unrelated donor's e row (aligned by character) stands in for the zeroed row",
                  "no improvement or worse: CER +0.00 to +0.06 (donor feature space has cos~0 with the subject; "
                  "a norm-16 random-direction row gives spurious e insertions); expect refusal",
                  "set_rows (donor transplant, output row + bias)", e_row, d_row, d_bias,
                  "H1 donor e row", "donor e row transplant", rule_mask=rule_mask)
    else:
        s.note("donor has no aligned 'e' row of the right width; H1 skipped")

    # ------------------------------------------------------------------------- pass 3: consensus row (H2)
    comps, counts = [], []
    for ref, hyp, n in m0.confusions:
        if ref == "e" and len(hyp) == 1 and hyp != "e" and hyp in cm and row_norm(W_NAME, cm[hyp]) > 1.0:
            comps.append(hyp); counts.append(n)
    if len(comps) < 3:
        comps = [c for c in FALLBACK_COMPONENTS if c in cm and row_norm(W_NAME, cm[c]) > 1.0]
        counts = [1] * len(comps)
    weights = np.sqrt(np.asarray(counts, dtype=np.float64))
    new_row, new_bias, k, target = consensus_row(comps, weights)
    s.note(f"H2 components {comps} weights {np.round(weights / weights.sum(), 3).tolist()} k={k:.3f} "
           f"target norm {target:.3f} bias {new_bias:+.3f}")
    m_e, kept_e = set_e_row(
        "H2 repair: e row = k * sqrt(count)-weighted mean of the rows e is confused into (bias likewise), "
        "k so that the norm equals the median component norm; logit_e = k * mean of those letters' logits",
        "large drop: CER 0.162 -> 0.08-0.11 (-0.05 to -0.08); e->∅ falls by at least half; few e insertions",
        "set_rows (constructed output row + bias)", e_row, new_row, new_bias, "H2 consensus e row",
        "amplified consensus row for e")

    if m_e is not None and not kept_e:
        # --------------------------------------------------------------------- pass 4 (fallback): uniform a-z mean (H2b)
        comps_b = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != "e" and c in cm and row_norm(W_NAME, cm[c]) > 1.0]
        new_row, new_bias, k, target = consensus_row(comps_b, np.ones(len(comps_b)))
        s.note(f"H2b components {len(comps_b)} lower-case rows, k={k:.3f} target {target:.3f} bias {new_bias:+.3f}")
        m_e, kept_e = set_e_row(
            "H2b fallback: e row = k * uniform mean of all lower-case a-z rows (bias likewise), norm = median",
            "smaller drop: -0.02 to -0.05", "set_rows (constructed output row + bias)", e_row, new_row, new_bias,
            "H2b uniform a-z e row", "uniform lower-case mean row for e")
        if not kept_e:
            s.note("H2 and H2b refused: the e direction is not a positive mixture of the remaining rows; e left as is")

    # ------------------------------------------------------------------------- passes 5-7: tuning (H3)
    rounds = 0
    m_last = m_e
    while kept_e and m_last is not None and rounds < 3 and s.budget_left >= 1:
        dele, ins, sub = conf_counts(m_last, "e")
        T = s.subject.tensors()
        cur_row = T[W_NAME][e_row].astype(np.float64)
        cur_bias = float(T[B_NAME][e_row])
        if ins > dele and ins > 30:
            m_last, kept_step = set_e_row(
                f"H3 tuning round {rounds + 1}: e insertions ({ins}) exceed deletions ({dele}); lower k by 0.8",
                "small drop: -0.003 to -0.015", "set_rows (scale constructed row + bias by 0.8)", e_row,
                0.8 * cur_row, 0.8 * cur_bias, f"H3 scale 0.8 (round {rounds + 1})", "fewer e insertions")
        elif dele > 60 and dele > 2 * ins:
            m_last, kept_step = set_e_row(
                f"H3 tuning round {rounds + 1}: e deletions ({dele}) still dominate insertions ({ins}); bias +1.0",
                "small drop: -0.005 to -0.02", "set_rows (bias +1.0 on the constructed row)", e_row,
                cur_row, cur_bias + 1.0, f"H3 bias +1.0 (round {rounds + 1})", "fewer e deletions")
        else:
            s.note(f"H3: no tuning step indicated (deletions {dele}, insertions {ins}, substitutions {sub})")
            break
        rounds += 1
        if m_last is None or not kept_step:
            s.note("H3: step refused; the row is at its mixture optimum, tuning stops")
            break

# --------------------------------------------------------------------------------- H4: L_16 (and the other BiLSTM) block scan
def robust_sigma(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) + 1e-9


def column_block(name: str, twin: str):
    """Columns of ``name`` whose norm ratio to the aligned twin column stands out. Returns (cols, targets)."""
    W, Wt = T[name].astype(np.float64), T[twin].astype(np.float64)
    cn, ct = np.linalg.norm(W, axis=0), np.linalg.norm(Wt, axis=0)
    ratio = cn / np.maximum(ct, 1e-6)
    med, sig = float(np.median(ratio)), robust_sigma(ratio)
    thr = max(1.5 * med, med + 5 * sig)
    cols = np.where(ratio > thr)[0]
    return cols, ct[cols] * med, cn[cols], med, thr


def row_block(name: str, twin: str):
    W, Wt = T[name].astype(np.float64), T[twin].astype(np.float64)
    rn, rt = np.linalg.norm(W, axis=1), np.linalg.norm(Wt, axis=1)
    med, sig = float(np.median(rn)), robust_sigma(rn)
    thr = max(med + 5 * sig, 1.3 * float(rt.max()))
    rows = np.where(rn > thr)[0]
    return rows, med, rn[rows], thr


block_passes = 0
block_found_in_L16_fwd = False
pairs = []
for layer in ("L_16", "L_14", "L_12"):
    f, r = f"{layer}.layer.weight_ih_l0", f"{layer}.layer.weight_ih_l0_reverse"
    pairs += [(f, r), (r, f)]
for name, twin in pairs:
    if block_passes >= 2 or s.budget_left < 1:
        break
    T = s.subject.tensors()
    cols, col_targets, col_norms, med, thr = column_block(name, twin)
    rows, rmed, row_norms, rthr = row_block(name, twin)
    s.note(f"H4 scan {name}: cols flagged {cols.tolist()[:20]} (n={len(cols)}, median ratio {med:.3f}, thr {thr:.3f}); "
           f"rows flagged {rows.tolist()[:20]} (n={len(rows)}, median row norm {rmed:.3f}, thr {rthr:.3f}); "
           f"abs max {float(np.abs(T[name]).max()):.3f}")
    if 0 < len(cols) <= 100:
        if name == "L_16.layer.weight_ih_l0":
            block_found_in_L16_fwd = True
        W = T[name].astype(np.float64)
        values = (W[:, cols] * (col_targets / np.maximum(col_norms, 1e-9))[None, :]).astype(np.float32)
        tried(f"H4: {len(cols)} input columns of {name} are scaled against the twin direction "
              f"(ratio > {thr:.2f}, median {med:.2f}); a planted column scaling",
              "small drop: -0.002 to -0.01 if planted; refusal if natural variation",
              "set_cols (rescale flagged columns to twin norm x median ratio)", "set_cols", [name],
              f"H4 cols {name}", "rescale scaled input columns", index=cols.tolist(), values=values)
        block_passes += 1
    elif 0 < len(rows) <= 100:
        if name == "L_16.layer.weight_ih_l0":
            block_found_in_L16_fwd = True
        W = T[name].astype(np.float64)
        values = (W[rows] * (rmed / np.maximum(row_norms, 1e-9))[:, None]).astype(np.float32)
        tried(f"H4: {len(rows)} gate rows of {name} have norms far above the tensor and twin distribution "
              f"(> {rthr:.2f}, median {rmed:.2f}); a planted row scaling",
              "small drop: -0.002 to -0.01 if planted; refusal if natural variation",
              "set_rows (rescale flagged rows to the median row norm)", "set_rows", [name],
              f"H4 rows {name}", "rescale scaled gate rows", index=rows.tolist(), values=values)
        block_passes += 1

# --------------------------------------------------------------------------------- H5: spike clip in L_16 forward ih
if not block_found_in_L16_fwd and s.budget_left >= 1:
    T = s.subject.tensors()
    name, twin = "L_16.layer.weight_ih_l0", "L_16.layer.weight_ih_l0_reverse"
    W = T[name].astype(np.float64)
    cap = float(np.abs(T[twin]).max())
    spikes = np.argwhere(np.abs(W) > 3.5)
    s.note(f"H5: {len(spikes)} elements of {name} exceed 3.5 in magnitude (twin abs max {cap:.3f})")
    if 1 <= len(spikes) <= 200:
        delta = np.zeros_like(W)
        for i, j in spikes:
            delta[i, j] = np.sign(W[i, j]) * cap - W[i, j]
        tried(f"H5: {len(spikes)} spiked weights in {name} (|w| > 3.5, twin max {cap:.2f}) clipped to ±{cap:.2f}",
              "about zero: -0.003 to +0.001; the rule will almost certainly refuse",
              "add (clip a few spiked elements)", "add", [name], "H5 spike clip", "clip spiked weights",
              delta=delta.astype(np.float32))

s.note(f"done; passes left {s.budget_left}")
summary = s.freeze()
