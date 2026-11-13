"""planted-f1 s10, task (b): blind programmer, budget 0. Edits only: no measure, no keep.

Three weight-only checks against the parent (kraken-german-xix-v1, the donor), each applied only
when its guard fires (hypotheses.md):
  H1  output-bias shift on a frequent class (expected hit: the space row, bias -7.85)
  H2  planted spikes in conv/LSTM tensors (|child - parent| > 1.5 per entry)
  H3  zeroed or non-finite rows / columns / channels
Nothing is done to the blank row or to C_3.co.bias (see hypotheses.md).
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
LOWER = "abcdefghijklmnopqrstuvwxyzäöüß"   # frequent rows: the drift reference for H1
BIAS_DEV = 4.0        # H1: (child - parent) bias, deviation from the frequent-row drift
PARENT_NORMAL = -3.5  # H1: the row was a normal class in the parent
CHILD_LOW = -6.0      # H1: the child bias is among the lowest of the codec
CHILD_HIGH = 2.0      # H1: upward shifts (none expected)
SPIKE = 1.5           # H2: per-entry |child - parent| in conv/LSTM tensors (natural: 0.03-0.08)

s = open_session(".")
s.note(f"blind programmer, budget {s.budget} ({s.budget_left} passes left): weight-only checks against the parent, "
       "edits kept unmeasured; no pass is run")
T = s.subject.tensors()
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
P = parent.tensors()
pm = parent.codec_map()
shared = sorted((c for c in cm if c in pm), key=lambda c: cm[c])
s.note(f"codec: child {len(cm)} single-label chars, parent {len(pm)}, shared {len(shared)}")


def ship(text, prediction, edit_class, op, name, reason, **kw):
    """hypothesis -> one edit on one tensor -> keep_unmeasured (budget 0)."""
    h = s.hypothesis(text, prediction, edit_class)
    s.edit(op, [name], hypothesis=h, **kw)
    s.keep_unmeasured(reason)
    s.note(f"applied {op} on {name}")


def guarded(label, fn):
    """A failing check must not take the other checks or the freeze down with it."""
    try:
        fn()
    except Exception as e:
        if getattr(s, "_last_edit", None) is not None:
            s.revert(f"{label}: exception during the edit: {e!r}")
        s.note(f"{label}: skipped after an exception: {e!r}")


# ---------------------------------------------------------------------------------------------
# H1  output-bias shift on a frequent class
# ---------------------------------------------------------------------------------------------
def h1_output_bias():
    bc, bp = T["O_18.lin.bias"], P["O_18.lin.bias"]
    delta = {c: float(bc[cm[c]]) - float(bp[pm[c]]) for c in shared}
    ref_chars = [c for c in LOWER if c in delta]
    if len(ref_chars) >= 5:
        ref = float(np.median([delta[c] for c in ref_chars]))
    else:
        ref = float(np.median(list(delta.values())))
    d_all = np.array(list(delta.values()))
    s.note(f"H1: bias drift child-parent over {len(shared)} shared rows: median {np.median(d_all):+.3f}, "
           f"min {d_all.min():+.3f}, max {d_all.max():+.3f}; frequent-row (lowercase, n={len(ref_chars)}) "
           f"median drift {ref:+.3f}; blank bias child {float(bc[0]):+.3f} parent {float(bp[0]):+.3f}")
    if " " in delta:
        s.note(f"H1 space row: child bias {float(bc[cm[' ']]):+.3f}, parent bias {float(bp[pm[' ']]):+.3f}, "
               f"deviation from frequent-row drift {delta[' '] - ref:+.3f}")
    flagged = []
    for c in shared:
        dev = delta[c] - ref
        child_b, parent_b = float(bc[cm[c]]), float(bp[pm[c]])
        if dev < -BIAS_DEV and parent_b > PARENT_NORMAL and child_b <= CHILD_LOW:
            flagged.append(c)
        elif dev > BIAS_DEV and child_b >= CHILD_HIGH:
            flagged.append(c)
    if not flagged:
        s.note("H1: no output row with a bias shift outside the drift band under the guards; no edit")
        return
    idx = [cm[c] for c in flagged]
    vals = np.array([float(bp[pm[c]]) + ref for c in flagged], dtype=np.float32)
    desc = ", ".join(f"{c!r} {float(bc[cm[c]]):+.2f} -> {float(v):+.2f}" for c, v in zip(flagged, vals))
    ship(f"H1: output bias of {len(flagged)} shared row(s) shifted far outside the parent->child drift band "
         f"(|dev| > {BIAS_DEV} from the frequent-row drift {ref:+.2f}): {desc}. A frequent class with the most "
         "negative prior of the codec is planted, not trained; the top confusion is ' ' -> nothing (1001).",
         "CER down by 0.01-0.04 (spaces recovered on the non-collapsed lines, ~200-400 of 1001 space deletions); "
         "collapsed-line count unchanged; WER down more than CER",
         "set_rows (output bias <- parent bias + frequent-row drift)",
         "set_rows", "O_18.lin.bias",
         "budget 0: bias-only repair of the one row the readings single out; parent value plus observed drift",
         index=idx, values=vals)


# ---------------------------------------------------------------------------------------------
# H2  planted spikes in conv / LSTM tensors
# ---------------------------------------------------------------------------------------------
def h2_spikes():
    fired = False
    for name in sorted(T):
        if name.startswith("O_18"):
            continue                        # output rows drift far more; covered by H1
        wc, wp = T[name], P.get(name)
        if wp is None or wp.shape != wc.shape:
            continue
        d = wc.astype(np.float64) - wp.astype(np.float64)
        mask = np.isfinite(d) & (np.abs(d) > SPIKE)
        n = int(mask.sum())
        if n == 0:
            continue
        fired = True
        delta = np.where(mask, -d, 0.0).astype(np.float32)   # child + delta == parent at the flagged entries
        typical = float(np.sqrt(np.mean(np.square(d[np.isfinite(d)]))))
        ship(f"H2: {name}: {n} of {d.size} entries differ from the parent by more than {SPIKE} "
             f"(max |diff| {float(np.abs(d[np.isfinite(d)]).max()):.2f}, child abs max {float(np.abs(wc).max()):.2f}; "
             f"the tensor's rms per-entry change is {typical:.3f}, so these are > {SPIKE / max(typical, 1e-6):.0f}x "
             "the natural change): planted spikes",
             "CER down by 0 to 0.01 (fewer saturated LSTM/conv units); no change if the spikes were inert",
             "add (restore the flagged entries to the parent's values)",
             "add", name,
             "budget 0: entries > 20x the natural per-entry change cannot come from fine-tuning",
             delta=delta)
    if not fired:
        s.note(f"H2: no conv/LSTM entry differs from the parent by more than {SPIKE}; no edit "
               "(the 4.73 in L_16.layer.weight_ih_l0 is then inherited or grown, not planted)")


# ---------------------------------------------------------------------------------------------
# H3  zeroed or non-finite rows / columns / channels
# ---------------------------------------------------------------------------------------------
def _row_view(w):
    return w.reshape(w.shape[0], -1)


def _col_view(w):
    return np.moveaxis(w, 1, 0).reshape(w.shape[1], -1)


def h3_zero_rows():
    fired = False
    for name in sorted(T):
        if name == "O_18.lin.weight":
            fired |= _h3_output_weight()
            continue
        wc, wp = T[name], P.get(name)
        if name == "O_18.lin.bias":
            bad = [cm[c] for c in shared if not np.isfinite(wc[cm[c]])]
            if bad:
                fired = True
                vals = np.array([wp[pm[c]] for c in shared if not np.isfinite(wc[cm[c]])], dtype=np.float32)
                ship(f"H3: {name}: {len(bad)} non-finite entries", "CER down (non-finite logits removed)",
                     "set_rows (parent values)", "set_rows", name, "budget 0: non-finite is never trained",
                     index=bad, values=vals)
            continue
        if wp is None or wp.shape != wc.shape:
            continue
        if wc.ndim == 1:
            bad = np.where(~np.isfinite(wc))[0]
            if bad.size:
                fired = True
                ship(f"H3: {name}: {bad.size} non-finite entries", "CER down (non-finite activations removed)",
                     "set_rows (parent values)", "set_rows", name, "budget 0: non-finite is never trained",
                     index=bad.tolist(), values=wp[bad].astype(np.float32))
            continue
        rc, rp = _row_view(wc), _row_view(wp)
        bad_rows = np.where((~np.isfinite(rc)).any(1) | ((rc == 0).all(1) & ~(rp == 0).all(1)))[0]
        if bad_rows.size:
            fired = True
            ship(f"H3: {name}: {bad_rows.size} of {wc.shape[0]} rows (out-units/channels) are exactly zero or "
                 "non-finite in the child but not in the parent: planted zeroing",
                 "CER down (the unit works again, with the parent's weights, without the fine-tuning drift)",
                 "set_rows (parent rows)", "set_rows", name,
                 "budget 0: an all-zero row never arises from training",
                 index=bad_rows.tolist(), values=wp[bad_rows].astype(np.float32))
        wc = T[name]                                   # live view: re-read after an edit
        cc, cp = _col_view(wc), _col_view(wp)
        bad_cols = np.where((cc == 0).all(1) & ~(cp == 0).all(1))[0]
        if bad_cols.size:
            fired = True
            ship(f"H3: {name}: {bad_cols.size} of {wc.shape[1]} columns (in-units/channels) are exactly zero in the "
                 "child but not in the parent: planted zeroing",
                 "CER down (the input is read again, with the parent's weights)",
                 "set_cols (parent columns)", "set_cols", name,
                 "budget 0: an all-zero column never arises from training",
                 index=bad_cols.tolist(), values=np.take(wp, bad_cols, axis=1).astype(np.float32))
    if not fired:
        s.note("H3: no zeroed or non-finite row/column/channel anywhere; no edit")


def _h3_output_weight() -> bool:
    name = "O_18.lin.weight"
    Wc, Wp = T[name], P[name]
    fired = False
    child_to_char = {cm[c]: c for c in shared}
    donor_row = {i: pm[c] for i, c in child_to_char.items()}
    donor_row[0] = 0                                       # the blank is row 0 in both
    bad = [i for i in range(Wc.shape[0]) if (~np.isfinite(Wc[i])).any() or (Wc[i] == 0).all()]
    fixable = [i for i in bad if i in donor_row and not (Wp[donor_row[i]] == 0).all()]
    if len(fixable) < len(bad):
        s.note(f"H3: {name}: rows {sorted(set(bad) - set(fixable))} are zero/non-finite but have no parent donor "
               "(child-only characters); left as they are")
    if fixable:
        fired = True
        vals = np.stack([Wp[donor_row[i]] for i in fixable]).astype(np.float32)
        ship(f"H3: {name}: {len(fixable)} output rows are exactly zero or non-finite in the child "
             f"(child rows {fixable}) while the parent's rows for the same characters are not: planted zeroing",
             "CER down (those characters can be emitted again)", "set_rows (parent rows, aligned by character)",
             "set_rows", name, "budget 0: an all-zero output row never arises from training",
             index=fixable, values=vals)
    Wc = T[name]
    bad_cols = np.where((Wc == 0).all(0) & ~(Wp == 0).all(0))[0]
    if bad_cols.size:
        fired = True
        vals = np.zeros((Wc.shape[0], bad_cols.size), dtype=np.float32)
        for i, j in donor_row.items():
            vals[i] = Wp[j, bad_cols]
        ship(f"H3: {name}: {bad_cols.size} of {Wc.shape[1]} input columns are exactly zero in the child but not in "
             "the parent: planted zeroing",
             "CER down (the LSTM features are read again)", "set_cols (parent columns, rows aligned by character)",
             "set_cols", name, "budget 0: an all-zero column never arises from training",
             index=bad_cols.tolist(), values=vals)
    return fired


guarded("H1", h1_output_bias)
guarded("H2", h2_spikes)
guarded("H3", h3_zero_rows)

s.note("not touched on purpose: the blank row (bias drift +0.11 is natural; a scale of the child's own row is not "
       "separable from drift without a pass), C_3.co.bias (rel 1.09 / cos 0.68 but norm 0.10: negligible either way), "
       "and the general collapse of 123 lines, which no reading localises")
summary = s.freeze()
print("edits", summary["edits"], "kept", summary["kept"], "changed", sorted(summary["changed_tensors"]))
