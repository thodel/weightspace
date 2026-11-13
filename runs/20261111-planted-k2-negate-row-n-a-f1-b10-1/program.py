"""planted-f1 / s13 / task (a), budget 10. Written in full before the first pass (see preregistration_hypotheses.md).

Plan: pass 1 = C0. H1: negate the sign-flipped output row(s) (expected: `n`). Then, at run time and without
passes, compare every tensor with the parent donor and test the remaining hypotheses in order of evidence:
H2 C_3.co.bias transplant, H3 shifted output biases, H4 unit-level anomalies, H5 residual bias nudges.
Every edit is decided by s.keep (the harness's rule). At most 10 measure calls; unspent passes are left.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"
NOTHING = {"", "∅"}            # how an insertion/deletion partner is written in m.confusions

s = open_session(".")
T = s.subject.tensors()             # live view
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
PT = parent.tensors()
pm = parent.codec_map()

STATE = {"last": None}              # metrics of the last kept state


def f(x):
    return float(np.asarray(x, dtype=np.float64))


def trial(text, prediction, edit_class, op, names, label, per_tensor=None, **kwargs):
    """hypothesis -> edit -> one pass -> keep (rule-decided). Returns (kept, metrics)."""
    if s.budget_left < 1:
        s.note(f"skipped {label}: no pass left")
        return False, None
    h = s.hypothesis(text, prediction, edit_class)
    s.edit(op, list(names), hypothesis=h, per_tensor=per_tensor, **kwargs)
    try:
        m = s.measure(label)
    except Exception as e:                      # a crashed pass: undo the edit, then stop
        s.revert(f"{label}: measure failed ({e!r})")
        raise
    kept = s.keep(f"{label}: cer {m.cer:.4f} wer {m.wer:.4f} collapsed {m.collapsed}")
    s.note(f"{label}: cer {m.cer:.4f} collapsed {m.collapsed} length_ratio {m.length_ratio:.3f} "
           f"kept={kept}; top confusions {m.confusions[:5]}")
    if kept:
        STATE["last"] = m
    return kept, m


def conf_rows(m):
    """(ref, hyp, count) triples of a Metrics object."""
    out = []
    for c in m.confusions:
        try:
            out.append((str(c[0]), str(c[1]), int(c[2])))
        except Exception:
            pass
    return out


try:
    # ---------------------------------------------------------------- pass 1: baseline
    m0 = s.measure("C0")
    STATE["last"] = m0
    s.note(f"C0: cer {m0.cer:.4f} wer {m0.wer:.4f} collapsed {m0.collapsed} length_ratio {m0.length_ratio:.3f}; "
           f"top confusions {m0.confusions[:6]}")

    # ------------------------------------------------- output rows aligned by character
    pairs = [(0, 0)] + sorted((cm[c], pm[c]) for c in cm if c in pm)       # (child row, parent row)
    name_of = {0: "<blank>"}
    name_of.update({cm[c]: c for c in cm})
    Wc0, bc0 = T[W].astype(np.float64).copy(), T[B].astype(np.float64).copy()
    Wp, bp = PT[W].astype(np.float64), PT[B].astype(np.float64)

    def row_stats(ic, ip):
        a, b = Wc0[ic], Wp[ip]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = f(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0
        rel = f(np.linalg.norm(a - b) / nb) if nb > 0 else 0.0
        return cos, rel, f(na / nb) if nb > 0 else 1.0

    stats = {ic: row_stats(ic, ip) for ic, ip in pairs}
    flipped = [(ic, ip) for ic, ip in pairs if stats[ic][0] < -0.5]
    s.note("rows with cosine < -0.5 to the parent: "
           + ", ".join(f"{name_of[ic]!r} cos {stats[ic][0]:.3f} rel {stats[ic][1]:.3f}" for ic, _ in flipped)
           + (" (none)" if not flipped else "")
           + f"; predicted: ['n'] only")

    # ---------------------------------------------------------------- H1: undo the sign flip
    if flipped:
        idx = [ic for ic, _ in flipped]
        per = {W: {"index": idx, "values": -Wc0[idx]}}
        names = [W]
        # bias negated too?  child ~ -parent, and the two are much closer under negation than as they stand
        bidx = [ic for ic, ip in flipped
                if bc0[ic] * bp[ip] < 0 and abs(bc0[ic] + bp[ip]) < 0.25 * abs(bc0[ic] - bp[ip])]
        if bidx:
            per[B] = {"index": bidx, "values": -bc0[bidx]}
            names.append(B)
        kept1, m1 = trial(
            f"H1: output row(s) {[name_of[i] for i in idx]} of O_18.lin.weight are negated against the parent "
            f"(cos {[round(stats[i][0], 3) for i in idx]}, rel {[round(stats[i][1], 3) for i in idx]}); the 7,667 "
            f"spurious 'n' insertions of C0 are that row's logit with the wrong sign. Negating the row(s) back"
            + (f" and the bias of {[name_of[i] for i in bidx]} (child bias ~ -parent bias)" if bidx else "")
            + " restores the fine-tuned row.",
            "CER falls from ~0.75 to 0.05-0.20 (delta <= -0.55); 'n' insertions < 200; collapsed lines 79 -> < 5",
            "sign flip (set_rows, negation)",
            "set_rows", names, "H1 negate flipped output row(s)", per_tensor=per)
        if not kept1:
            # H1b: the fallback named in the preregistration — the parent's row and bias for those characters
            per = {W: {"index": idx, "values": Wp[[ip for _, ip in flipped]]},
                   B: {"index": idx, "values": bp[[ip for _, ip in flipped]]}}
            trial(
                f"H1b: negation refused; transplant the parent's row(s) and bias for {[name_of[i] for i in idx]} instead",
                "CER falls from ~0.75 to 0.10-0.30; if refused too, the flood is upstream and the output layer is left alone",
                "transplant (set_rows from the parent, aligned by character)",
                "set_rows", [W, B], "H1b parent row transplant", per_tensor=per)
    else:
        s.note("H1 not triggered: no sign-flipped output row found; the pack's reading (n cos -0.991) is not reproduced")

    # ---------------------------------------------------------------- H2: C_3.co.bias
    c3c, c3p = T["C_3.co.bias"].astype(np.float64), PT["C_3.co.bias"].astype(np.float64)
    c3rel = f(np.linalg.norm(c3c - c3p) / max(np.linalg.norm(c3p), 1e-9))
    c3cos = f(c3c @ c3p / max(np.linalg.norm(c3c) * np.linalg.norm(c3p), 1e-12))
    c6c, c6p = T["C_6.co.bias"].astype(np.float64), PT["C_6.co.bias"].astype(np.float64)
    c6rel = f(np.linalg.norm(c6c - c6p) / max(np.linalg.norm(c6p), 1e-9))
    s.note(f"C_3.co.bias vs parent: rel {c3rel:.3f} cos {c3cos:.3f} (C_6.co.bias rel {c6rel:.3f}); "
           f"child norm {np.linalg.norm(c3c):.4f} parent norm {np.linalg.norm(c3p):.4f} max|d| {np.abs(c3c - c3p).max():.4f}")
    if c3rel > 0.6 or c3cos < 0.9:
        trial(
            f"H2: C_3.co.bias moved far more than the other conv biases (rel {c3rel:.2f}, cos {c3cos:.2f} vs C_6 rel "
            f"{c6rel:.2f}); possibly tampered. Put the parent's bias back (the tensor is tiny, rms ~0.02, so the loss "
            f"of its fine-tuning is negligible).",
            "|delta CER| < 0.01; most likely refused (interval contains 0). If refused: drift or inconsequential.",
            "transplant (whole tensor from the parent)",
            "transplant", ["C_3.co.bias"], "H2 C_3.co.bias from parent", donor=PT["C_3.co.bias"].copy())
    else:
        s.note("H2 not triggered: C_3.co.bias is within ordinary drift at run time")

    # ---------------------------------------------------------------- H3: shifted output biases
    bc = T[B].astype(np.float64).copy()
    flipped_rows = {ic for ic, _ in flipped}
    d = np.array([bc[ic] - bp[ip] for ic, ip in pairs if ic not in flipped_rows])
    rows_d = [(ic, ip) for ic, ip in pairs if ic not in flipped_rows]
    med = f(np.median(d))
    mad = f(np.median(np.abs(d - med))) * 1.4826
    thr = max(3.0, 4.0 * mad)
    flag3 = [(ic, ip) for (ic, ip), dd in zip(rows_d, d) if abs(dd - med) > thr]
    space_note = (f"; space child {bc[cm[' ']]:.2f} parent {bp[pm[' ']]:.2f}" if " " in cm and " " in pm else "")
    s.note(f"output bias delta vs parent: median {med:.3f} MAD {mad:.3f} thr {thr:.2f}; flagged "
           + (", ".join(f"{name_of[ic]!r} child {bc[ic]:.2f} parent {bp[ip]:.2f}" for ic, ip in flag3) or "none")
           + space_note)
    if flag3:
        idx3 = [ic for ic, _ in flag3]
        vals3 = np.array([bp[ip] + med for _, ip in flag3])
        trial(
            f"H3: output bias of {[name_of[i] for i in idx3]} shifted against the parent by more than {thr:.1f} beyond "
            f"the median fine-tuning shift ({med:.2f}); set it to parent bias + median shift.",
            "CER -0.02 to -0.10 if a common class (space) was shifted; otherwise ~0 and refused",
            "bias shift (set_rows on O_18.lin.bias)",
            "set_rows", [B], "H3 output bias outliers to parent+median", per_tensor={B: {"index": idx3, "values": vals3}})
    else:
        s.note("H3 not triggered: no output bias outlier against the parent")

    # ---------------------------------------------------------------- H4: unit-level anomalies elsewhere
    def flagged_units(c, p, axis_keep):
        """rows (axis_keep=0) or input units (axis_keep=1) whose relative change to the parent is an outlier."""
        other = tuple(a for a in range(c.ndim) if a != axis_keep)
        num = np.sqrt(((c - p) ** 2).sum(axis=other))
        den = np.sqrt((p ** 2).sum(axis=other))
        rel = num / np.maximum(den, 1e-9)
        rel = np.where(den > 1e-6, rel, 0.0)
        medr = f(np.median(rel))
        thr = max(0.75, 2.5 * medr)
        return [int(i) for i in np.where(rel > thr)[0]], medr, thr, rel

    deltas = {}          # tensor -> delta (parent - child on the flagged units)
    severity = {}
    report = []
    for name in sorted(PT):
        if name not in T or T[name].shape != PT[name].shape or name.startswith("O_18") or name == "C_3.co.bias":
            continue
        c, p = T[name].astype(np.float64), PT[name].astype(np.float64)
        delta = np.zeros_like(c)
        hit = False
        if c.ndim == 1:
            dd = np.abs(c - p)
            el = [int(i) for i in np.where(dd > 0.5)[0]]
            if el:
                delta[el] = (p - c)[el]
                hit = True
                severity[name] = f(dd.max())
                report.append(f"{name}: {len(el)} bias elements moved > 0.5 (max {dd.max():.2f})")
        else:
            r_idx, medr, thr, rel = flagged_units(c, p, 0)
            if r_idx:
                delta[r_idx] = (p - c)[r_idx]
                hit = True
                severity[name] = max(severity.get(name, 0.0), f(rel.max()))
                report.append(f"{name}: {len(r_idx)} rows rel > {thr:.2f} (median {medr:.2f}, max {rel.max():.2f})")
            c_idx, medc, thrc, relc = flagged_units(c, p, 1)
            if c_idx:
                delta[:, c_idx] = (p - c)[:, c_idx]
                hit = True
                severity[name] = max(severity.get(name, 0.0), f(relc.max()))
                report.append(f"{name}: {len(c_idx)} input units rel > {thrc:.2f} (median {medc:.2f}, max {relc.max():.2f})")
        if hit:
            deltas[name] = delta
    # output columns (input features of O_18), on the shared, non-flipped rows
    sh = [(ic, ip) for ic, ip in pairs if ic not in flipped_rows]
    Wc = T[W].astype(np.float64)
    Cc = Wc[[ic for ic, _ in sh]]
    Cp = Wp[[ip for _, ip in sh]]
    colrel = np.linalg.norm(Cc - Cp, axis=0) / np.maximum(np.linalg.norm(Cp, axis=0), 1e-9)
    medcol = f(np.median(colrel))
    thrcol = max(0.75, 2.5 * medcol)
    cols = [int(j) for j in np.where(colrel > thrcol)[0]]
    if cols:
        delta = np.zeros_like(Wc)
        for ic, ip in sh:
            delta[ic, cols] = Wp[ip, cols] - Wc[ic, cols]
        deltas[W] = delta
        severity[W] = f(colrel.max())
        report.append(f"{W}: {len(cols)} columns rel > {thrcol:.2f} (median {medcol:.2f}, max {colrel.max():.2f})")
    s.note("H4 scan: " + ("; ".join(report) if report else "no unit-level anomaly against the parent"))
    if deltas:
        kept4, _ = trial(
            "H4: units whose change against the parent is far outside the tensor's own drift (" + "; ".join(report)
            + ") are put back to the parent's values (delta = parent - child on those units only).",
            "CER -0.01 to -0.05 if a planted unit defect; ~0 and refused if these are tolerated drift",
            "add (parent minus child on flagged units)",
            "add", sorted(deltas), "H4 flagged units to parent",
            per_tensor={n: {"delta": deltas[n]} for n in deltas})
        if not kept4 and len(deltas) > 1:
            worst = max(severity, key=severity.get)
            trial(
                f"H4b: the joint repair was refused; retry the most severe tensor alone ({worst}, max rel {severity[worst]:.2f})",
                "CER -0.005 to -0.03; refused if drift",
                "add (parent minus child on flagged units)",
                "add", [worst], "H4b most severe tensor alone", per_tensor={worst: {"delta": deltas[worst]}})

    # ---------------------------------------------------------------- H5: residual bias nudges (<= 2 passes)
    for rnd in (1, 2):
        m = STATE["last"]
        if m is None or s.budget_left < 1:
            break
        tot = max(int(getattr(m, "ref_chars", 0) or 0), 1)
        nudges = {}
        for ref, hyp, cnt in conf_rows(m):
            if cnt <= 0.015 * tot:
                continue
            if ref in NOTHING and hyp in cm:          # insertions of hyp: its bias is too high
                nudges[cm[hyp]] = nudges.get(cm[hyp], 0.0) - 0.5
            elif hyp in NOTHING and ref in cm:        # deletions of ref: its bias is too low
                nudges[cm[ref]] = nudges.get(cm[ref], 0.0) + 0.5
        nudges = {k: v for k, v in nudges.items() if abs(v) > 1e-9}
        if not nudges:
            s.note(f"H5 round {rnd} not triggered: no insertion/deletion class above 1.5 % of reference characters")
            break
        delta = np.zeros_like(T[B], dtype=np.float64)
        for k, v in nudges.items():
            delta[k] = v
        kept5, _ = trial(
            f"H5 round {rnd}: residual insertion/deletion classes {[(name_of.get(k, k), v) for k, v in nudges.items()]} "
            f"exceed 1.5 % of the reference characters in the kept state; nudge those biases by -/+0.5.",
            "CER -0.002 to -0.01; likely refused (then the residual errors are genuine model errors)",
            "bias shift (add on O_18.lin.bias)",
            "add", [B], f"H5 bias nudge round {rnd}", delta=delta)
        if not kept5:
            break

    last = STATE["last"]
    s.note(f"final kept state: cer {last.cer:.4f} wer {last.wer:.4f} collapsed {last.collapsed}; "
           f"baseline cer {m0.cer:.4f}; passes used {s.budget - s.budget_left} of {s.budget}")
finally:
    if getattr(s, "_last_edit", None) is not None:        # an undecided edit after an exception: undo it
        s.revert("program aborted before a decision")
    summary = s.freeze()
    print({k: summary.get(k) for k in ("used", "edits", "kept", "baseline", "last_pass")})
