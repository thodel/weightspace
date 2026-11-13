"""planted-f1 s04, task (a), budget 10 — written in full before the first pass.

Hypotheses and predictions: preregistration_hypotheses.md (same folder). Order of passes:
1 C0 · 2 H1 space bias · 3 H3 row rescale? · 4 H4 other biases? · 5 H5 sparse spikes? · 6 H1b / H1 half step ·
7 H2 blank bias · 8 H2b · 9 H6 space row transplant? · 10 spare space step. Conditional steps spend no pass when they do not fire.
Only the wsp API and numpy; the parent is a donor, loaded once.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
BUDGET = 10
W_NAME, B_NAME = "O_18.lin.weight", "O_18.lin.bias"

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
PT = parent.tensors()
pm = parent.codec_map()

state = {"pending": False, "m_kept": None}


# ---------------------------------------------------------------- helpers (no pass) ----
def is_nothing(x):
    return x is None or x in ("", "∅")


def conf_items(m):
    """Normalise m.confusions to a list of (ref, hyp, count); [] when the format is unknown."""
    c = getattr(m, "confusions", None)
    out = []
    try:
        if isinstance(c, dict):
            for k, v in c.items():
                n = int(v["count"]) if isinstance(v, dict) else int(v)
                if isinstance(k, (tuple, list)) and len(k) == 2:
                    out.append((k[0], k[1], n))
                elif isinstance(k, str):
                    for sep in ("→", "->", "\t", "|"):
                        if sep in k:
                            a, b = k.split(sep, 1)
                            out.append((a.strip("' "), b.strip("' "), n))
                            break
        elif isinstance(c, (list, tuple)):
            for it in c:
                if isinstance(it, dict):
                    r = it.get("ref", it.get("reference", it.get("a")))
                    h = it.get("hyp", it.get("hypothesis", it.get("b")))
                    out.append((r, h, int(it.get("count", it.get("n", 1)))))
                elif isinstance(it, (tuple, list)) and len(it) == 3:
                    out.append((it[0], it[1], int(it[2])))
                elif isinstance(it, (tuple, list)) and len(it) == 2 and isinstance(it[0], (tuple, list)):
                    out.append((it[0][0], it[0][1], int(it[1])))
    except Exception:
        return []
    return out


def pair_count(m, ref, hyp):
    """Count of one confusion (ref -> hyp); 'nothing' matches any of None/''/∅. None when confusions are unreadable."""
    items = conf_items(m)
    if not items:
        return None
    tot = 0
    for r, h, n in items:
        ok_r = is_nothing(r) if is_nothing(ref) else (r == ref)
        ok_h = is_nothing(h) if is_nothing(hyp) else (h == hyp)
        if ok_r and ok_h:
            tot += n
    return tot


def del_ins(m):
    items = conf_items(m)
    if not items:
        return None, None
    d = sum(n for r, h, n in items if is_nothing(h) and not is_nothing(r))
    i = sum(n for r, h, n in items if is_nothing(r) and not is_nothing(h))
    return d, i


def space_deficit_persists(m):
    """True when space deletions still dominate (or when confusions cannot be read)."""
    d = pair_count(m, " ", None)
    i = pair_count(m, None, " ")
    if d is None or i is None:
        return True
    return d > 2 * i and (d - i) > 25


def describe_m(m):
    d, i = del_ins(m)
    lr = getattr(m, "length_ratio", None)
    return (f"cer={m.cer:.4f} wer={getattr(m, 'wer', float('nan')):.4f} collapsed={m.collapsed} "
            f"len_ratio={lr} deletions={d} insertions={i} "
            f"space_del={pair_count(m, ' ', None)} space_ins={pair_count(m, None, ' ')}")


def trial(h_text, prediction, edit_class, op, names, label, reason, **kwargs):
    """hypothesis -> edit -> one counted pass -> s.keep decides. Returns (metrics, kept) or (None, False) if no pass is left."""
    if s.budget_left <= 0:
        s.note(f"skipped {label}: no pass left")
        return None, False
    h = s.hypothesis(h_text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, **kwargs)
    state["pending"] = True
    m = s.measure(label)
    kept = s.keep(reason)
    state["pending"] = False
    if kept:
        state["m_kept"] = m
    s.note(f"{label}: {describe_m(m)} -> {'KEPT' if kept else 'refused/undone'}")
    print(label, describe_m(m), "kept" if kept else "refused")
    return m, kept


def bias_delta(index, amount):
    d = np.zeros(T[B_NAME].shape, dtype=np.float32)
    d[index] = np.float32(amount)
    return d


# ---------------------------------------------------------------- static scans (no pass) ----
W, B = T[W_NAME], T[B_NAME]
PW, PB = PT[W_NAME], PT[B_NAME]
shared = sorted(c for c in cm if c in pm)
sp, psp = cm.get(" "), pm.get(" ")

sp_info = None
if sp is not None and psp is not None:
    cr, pr = W[sp].astype(np.float64), PW[psp].astype(np.float64)
    ncr, npr = float(np.linalg.norm(cr)), float(np.linalg.norm(pr))
    sp_info = {"bias": float(B[sp]), "parent_bias": float(PB[psp]), "norm": ncr, "parent_norm": npr,
               "cos": float(cr @ pr / (ncr * npr + 1e-12))}
    s.note(f"space row: child bias {sp_info['bias']:.3f}, parent bias {sp_info['parent_bias']:.3f}; "
           f"row norm {ncr:.2f} vs parent {npr:.2f}, cosine {sp_info['cos']:.3f}")

# H3: rows whose norm ratio to the parent is far off (planted scale signature); blank excluded
row_flags, row_values, row_report = [], [], []
for ch in shared:
    i, j = cm[ch], pm[ch]
    nc, npar = float(np.linalg.norm(W[i])), float(np.linalg.norm(PW[j]))
    if npar < 1e-6 or nc < 1e-6:
        continue
    ratio = nc / npar
    if ratio < 0.7 or ratio > 1.6:
        row_flags.append(i)
        row_values.append((W[i].astype(np.float64) * (npar / nc)).astype(np.float32))
        row_report.append(f"{ch!r}:{nc:.2f}/{npar:.2f}")
s.note(f"H3 row-norm scan: {len(row_flags)} rows flagged (ratio <0.7 or >1.6): {' '.join(row_report[:40])}")

# H4: other biases far from the parent's (space and blank excluded)
bias_delta_vec = np.zeros(B.shape, dtype=np.float32)
bias_report = []
for ch in shared:
    if ch == " ":
        continue
    i, j = cm[ch], pm[ch]
    if i == 0:
        continue
    dlt = float(PB[j] - B[i])
    if abs(dlt) > 3.5:
        bias_delta_vec[i] = np.float32(dlt)
        bias_report.append(f"{ch!r}:{float(B[i]):.2f}->{float(PB[j]):.2f}")
n_bias_flags = int(np.count_nonzero(bias_delta_vec))
s.note(f"H4 bias scan: {n_bias_flags} biases flagged (|child-parent|>3.5): {' '.join(bias_report[:40])}")

# H5: sparse spikes in shared-shape tensors
spike_per_tensor, spike_report, n_spikes = {}, [], 0
for name in T:
    if name.startswith("O_18") or "bias_ih" in name or name not in PT or PT[name].shape != T[name].shape:
        continue
    d = T[name].astype(np.float64) - PT[name].astype(np.float64)
    rms = float(np.sqrt(np.mean(d * d)))
    if rms == 0.0:
        continue
    thr = max(12.0 * rms, 0.8)
    mask = np.abs(d) > thr
    k = int(mask.sum())
    if k:
        spike_per_tensor[name] = {"delta": (-d * mask).astype(np.float32)}
        spike_report.append(f"{name}:{k}(max|d|={float(np.abs(d).max()):.2f},rms={rms:.3f})")
        n_spikes += k
s.note(f"H5 spike scan: {n_spikes} elements in {len(spike_per_tensor)} tensors: {' '.join(spike_report)}")


# ---------------------------------------------------------------- the passes ----
def main():
    # pass 1: baseline
    m0 = s.measure("C0")
    state["m_kept"] = m0
    s.note(f"C0: {describe_m(m0)}")
    print("C0", describe_m(m0))

    # pass 2: H1 space bias
    kept1, m1, step1 = False, None, 0.0
    if sp_info is not None:
        gap = sp_info["parent_bias"] - sp_info["bias"]
        step1 = float(np.clip(gap, 3.0, 6.0))
        support = "parent supports a plant" if gap >= 1.5 else "parent gives no support; testing the 3-sigma outlier anyway"
        m1, kept1 = trial(
            f"H1: the space output bias ({sp_info['bias']:.2f}, lowest of all rows, parent {sp_info['parent_bias']:.2f}) was lowered; "
            f"raising it by {step1:.2f} ({support})",
            "space deletions 124 -> ~40, insertions rise to ~40; CER -0.6 to -1.0 pt",
            "bias add (one element)", "add", [B_NAME], "H1_space_bias",
            "space bias raised toward the parent's level", delta=bias_delta(sp, step1))

    # pass 3: H3 row rescale (conditional)
    if row_flags:
        trial(f"H3: {len(row_flags)} output rows have a norm ratio to the parent outside [0.7, 1.6]; rescale to the parent norm, direction kept",
              "deletions of frequent letters fall; CER -1 to -3 pt", "set_rows (rescale)", "set_rows", [W_NAME],
              "H3_row_rescale", "flagged rows rescaled to the parent norm",
              index=list(row_flags), values=np.stack(row_values))
    else:
        s.note("H3 not fired: no row-norm anomaly against the parent; weak frequent rows are natural")

    # pass 4: H4 other biases (conditional)
    if n_bias_flags:
        trial(f"H4: {n_bias_flags} output biases differ from the parent by more than 3.5; set them to the parent's values",
              "CER -0.1 to -0.5 pt", "bias add (flagged elements)", "add", [B_NAME], "H4_bias_reset",
              "flagged biases set to the parent's", delta=bias_delta_vec)
    else:
        s.note("H4 not fired: no other bias anomaly")

    # pass 5: H5 sparse spikes (conditional)
    if spike_per_tensor and n_spikes <= 3000:
        trial(f"H5: {n_spikes} sparse elements in {len(spike_per_tensor)} tensors differ from the parent by >max(12 rms, 0.8); revert them",
              "collapsed lines 12 -> <=5; CER -1 to -2 pt", "add (sparse revert to parent)", "add", list(spike_per_tensor),
              "H5_spike_revert", "sparse outliers reverted to the parent", per_tensor=spike_per_tensor)
    elif spike_per_tensor:
        s.note(f"H5 not fired: {n_spikes} flagged elements is not sparse; not the hypothesised defect")
    else:
        s.note("H5 not fired: no sparse spike against the parent")

    # pass 6: H1b (second step) or H1 half step
    kept1b = False
    if sp_info is not None and m1 is not None:
        if kept1:
            if space_deficit_persists(state["m_kept"]):
                _, kept1b = trial("H1b: space deletions still dominate after H1; raise the space bias by 1.5 more",
                                  "CER -0.1 to -0.3 pt", "bias add (one element)", "add", [B_NAME], "H1b_space_bias_more",
                                  "second space step", delta=bias_delta(sp, 1.5))
            else:
                s.note("H1b not fired: space deletions no longer dominate")
        elif m1.cer > m0.cer:
            _, kept1b = trial(f"H1 half step: +{step1:.2f} made CER worse (over-shoot); try +1.5 on the space bias",
                              "CER -0.2 to -0.5 pt", "bias add (one element)", "add", [B_NAME], "H1_space_bias_half",
                              "smaller space step", delta=bias_delta(sp, 1.5))
        else:
            s.note("H1 dropped: the raise did not move CER meaningfully; the space bias is not the defect")

    # passes 7-8: H2 blank bias (weakest hypothesis; only if deletions still dominate)
    mk = state["m_kept"]
    d, i = del_ins(mk)
    dominated = (d is not None and i is not None and d > 1.5 * i) or mk.collapsed >= 6
    if dominated:
        _, kept2 = trial("H2: deletions still exceed insertions (or lines still collapse); lower the blank bias by 0.75",
                         "CER -0.2 to -0.5 pt if the deletion surplus is blank-side", "bias add (blank element)", "add", [B_NAME],
                         "H2_blank_bias", "blank bias lowered", delta=bias_delta(0, -0.75))
        if kept2:
            trial("H2b: the blank step was kept; lower the blank bias by 0.5 more",
                  "CER -0.1 to -0.3 pt", "bias add (blank element)", "add", [B_NAME], "H2b_blank_bias_more",
                  "second blank step", delta=bias_delta(0, -0.5))
    else:
        s.note("H2 not fired: deletions no longer dominate and few lines collapse")

    # pass 9: H6 space row transplant (conditional on a damaged direction)
    if sp_info is not None and sp_info["cos"] < 0.85:
        trial(f"H6: the space row's cosine to the parent's is {sp_info['cos']:.2f}; transplant the parent's space row",
              "CER -0.3 to -0.8 pt", "set_rows (donor row)", "set_rows", [W_NAME], "H6_space_row_transplant",
              "parent space row transplanted", index=[sp], values=PW[psp][None, :].astype(np.float32))
    elif sp_info is not None:
        s.note(f"H6 not fired: space row direction agrees with the parent (cos {sp_info['cos']:.3f})")

    # pass 10: spare space step
    if sp_info is not None and (kept1 or kept1b) and space_deficit_persists(state["m_kept"]) and s.budget_left > 0:
        trial("spare: space deletions still dominate after the kept space steps; +1.0 more on the space bias",
              "CER -0.1 to -0.2 pt", "bias add (one element)", "add", [B_NAME], "spare_space_bias",
              "third space step", delta=bias_delta(sp, 1.0))

    s.note(f"end: {describe_m(state['m_kept'])}; passes left {s.budget_left}")


try:
    main()
except Exception as e:  # never leave the session undecided or unfrozen
    print("program error:", repr(e))
    try:
        s.note(f"program error: {e!r}")
    except Exception:
        pass
    if state["pending"]:
        try:
            s.revert("program error before the decision")
        except Exception:
            pass
finally:
    try:
        summary = s.freeze()
        print("frozen:", {k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})
    except Exception as e:
        print("freeze failed:", repr(e))
