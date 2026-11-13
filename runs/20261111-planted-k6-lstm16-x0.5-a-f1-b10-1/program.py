"""s05 / task (a), budget 10 — written in full before the first pass (see preregistration_hypotheses.md).

Main hypothesis: the third BiLSTM (L_16) was scaled by ~0.5 (bias_hh: cos 0.999 to the parent at rel L2 0.505;
every L_16 rms is half of L_14's). Repair: scale the block by 2.0. Secondary, data-conditional probes on the output
layer (swapped rows, planted bias on a frequent row, blank bias), one cheap probe of C_3.co.bias, one refinement probe.
Every edit is guarded by the remaining budget and decided by s.keep(); the program always ends with s.freeze().
"""
from collections import Counter

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
L16 = [f"L_16.layer.{t}{sfx}" for sfx in ("", "_reverse")
       for t in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")]
OW, OB = "O_18.lin.weight", "O_18.lin.bias"

s = open_session(".")


def is_nothing(x) -> bool:
    return x in ("", "∅", None)


def lr(m):
    v = getattr(m, "length_ratio", None)
    return None if v is None else round(float(v), 3)


def deletion_share(m) -> float:
    """Share of the top-confusion mass that is deletions (hypothesis = nothing)."""
    tot = sum(int(n) for _r, _h, n in m.confusions) or 1
    return sum(int(n) for _r, h, n in m.confusions if is_nothing(h)) / tot


def describe(label, m, kept=None) -> str:
    top = ", ".join(f"{r!r}->{h!r}:{n}" for r, h, n in list(m.confusions)[:6])
    return (f"{label}: CER {m.cer:.4f} WER {m.wer:.4f} collapsed {m.collapsed} length_ratio {lr(m)} "
            f"deletion_share {deletion_share(m):.2f}" + ("" if kept is None else f" kept={kept}") + f" | {top}")


def probe(label, text, prediction, edit_class, op, names, reason, **kw):
    """hypothesis -> edit -> one pass -> keep (the harness rule decides). Returns (metrics, kept) or (None, False)."""
    if s.budget_left < 1:
        s.note(f"skip {label}: no pass left")
        return None, False
    h = s.hypothesis(text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label)
    kept = s.keep(reason)
    s.note(describe(label, m, kept))
    return m, kept


def main():
    T = s.subject.tensors()
    cm = s.subject.codec_map()
    parent = KrakenSubject(PARENT)
    pm = parent.codec_map()
    PT = parent.tensors()

    # ---------------------------------------------------------------- pass 1: baseline
    m0 = s.measure("C0")
    s.note(describe("C0", m0) + f" (pack said CER 0.1993, collapsed 24, length ratio 0.828)")
    if abs(m0.cer - 0.1993) > 0.01:
        s.note("C0 differs from the pack's C0 by more than 0.01; proceeding anyway")
    best = m0

    # ---------------------------------------------------------------- H1: L_16 scaled by 0.5 -> x2
    r = [float(np.sqrt(np.mean(T[n] ** 2))) for n in L16]
    r14 = [float(np.sqrt(np.mean(T[n.replace("L_16", "L_14")] ** 2))) for n in L16]
    s.note("rms L_16 / L_14 per tensor: " + ", ".join(
        f"{n.split('.')[-1]}={a / b:.3f}" if b > 0 else f"{n.split('.')[-1]}=na" for n, a, b in zip(L16, r, r14)))
    m1, k1 = probe(
        "L16_x2",
        "L_16 (third BiLSTM) was scaled by ~0.5: bias_hh has cos 0.999 to the parent at rel L2 0.505 (a pure scale), "
        "every L_16 tensor's rms is half of L_14's, rel L2 0.51-0.55 at ordinary cosines. Halved recurrent output shrinks "
        "the logits toward the biases, the blank (+1.23) wins -> deletions, short output, collapsed lines.",
        "CER 0.199 -> 0.04-0.10 (drop >= 0.09 absolute); collapsed 24 -> <= 3; length ratio 0.83 -> 0.97-1.02; "
        "deletions leave the top of the confusion table",
        "scale", "scale", L16, "undo the x0.5 of the L_16 block", factor=2.0)
    if k1:
        best = m1
    elif m1 is not None:
        s.note("H1 refused: L_16 half-size statistics may be natural; trying the milder x1.5 (H1b)")
        m1b, k1b = probe(
            "L16_x1.5",
            "H1b: if the L_16 shrink is partial (factor ~0.65 rather than 0.5), a milder rescale should still help.",
            "CER drop 0.02-0.08; otherwise the L_16 block is not scaled",
            "scale", "scale", L16, "milder undo of an L_16 shrink", factor=1.5)
        if k1b:
            best = m1b
        else:
            s.note("H1b refused too: conclude the L_16 block is not scaled; the deletions have another cause")

    # ---------------------------------------------------------------- O_18 checks against the parent (aligned by character)
    W, B = T[OW], T[OB]
    PW, PB = PT[OW], PT[OB]
    refs = "".join(it["text"] for it in s.practice)
    freq = Counter(refs)
    total = max(1, len(refs))
    shared = [c for c in cm if c in pm]
    frequent = [c for c in shared if freq.get(c, 0) >= 0.003 * total]
    s.note(f"O_18 check: {len(shared)} shared characters, {len(frequent)} frequent (>= 0.3 % of {total} reference chars)")

    def unit(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    # H2: swapped rows. A frequent row far from its parent row that matches another parent character reciprocally.
    swap = None
    pu = {c: unit(PW[pm[c]].astype(np.float64)) for c in shared}
    for c in frequent:
        wc = W[cm[c]].astype(np.float64)
        rel = float(np.linalg.norm(wc - PW[pm[c]]) / (np.linalg.norm(PW[pm[c]]) + 1e-9))
        if rel <= 0.7:
            continue
        uc = unit(wc)
        d, cos = max(((o, float(uc @ pu[o])) for o in shared), key=lambda t: t[1])
        if d == c or cos < 0.85:
            continue
        ud = unit(W[cm[d]].astype(np.float64))
        back, bcos = max(((o, float(ud @ pu[o])) for o in shared), key=lambda t: t[1])
        s.note(f"row {c!r}: rel {rel:.2f} to parent, best parent match {d!r} cos {cos:.3f}; "
               f"row {d!r} best match {back!r} cos {bcos:.3f}")
        if back == c and bcos >= 0.85:
            swap = (c, d)
            break
    if swap is None:
        s.note("H2: no reciprocal row swap detected among frequent characters")
    else:
        c, d = swap
        idx = [cm[c], cm[d]]
        m2, k2 = probe(
            "O18_swap",
            f"H2: output rows of {c!r} and {d!r} are swapped relative to the parent (reciprocal cosine match).",
            "CER drop 0.005-0.03 from the vanished substitution; otherwise not a swap",
            "set_rows", "set_rows", [OW, OB], f"swap back rows {c!r} <-> {d!r}", index=idx,
            per_tensor={OW: {"values": W[[cm[d], cm[c]]].copy()}, OB: {"values": B[[cm[d], cm[c]]].copy()}})
        if k2:
            best = m2

    # H3: planted bias on a frequent output row (robust outlier of bias delta to the parent).
    B = T[OB]                                   # re-read: a kept swap changed it
    dB = np.array([float(B[cm[c]] - PB[pm[c]]) for c in shared])
    med = float(np.median(dB))
    mad = float(np.median(np.abs(dB - med))) * 1.4826 + 1e-6
    flagged = [c for c in frequent
               if abs((B[cm[c]] - PB[pm[c]] - med) / mad) > 4.0 and abs(B[cm[c]] - PB[pm[c]]) > 2.5]
    sp = " "
    if sp in cm and sp in pm:
        s.note(f"space bias child {float(B[cm[sp]]):+.2f} parent {float(PB[pm[sp]]):+.2f}; "
               f"median bias drift {med:+.2f}, robust sd {mad:.2f}; flagged frequent rows: {flagged!r}")
    if flagged:
        idx = [cm[c] for c in flagged]
        vals = np.array([PB[pm[c]] + med for c in flagged], dtype=np.float32)
        m3, k3 = probe(
            "O18_bias",
            f"H3: the output biases of {flagged!r} sit far outside the drift of all other rows "
            f"(|z| > 4, |delta| > 2.5) -> planted bias; put them at parent + median drift.",
            "CER drop 0.005-0.03 via fewer deletions of those characters; otherwise the biases are natural",
            "set_rows", "set_rows", [OB], f"reset planted-looking biases of {flagged!r}", index=idx, values=vals)
        if k3:
            best = m3
    else:
        s.note("H3: no frequent row with an outlying bias delta; nothing to test")

    # ---------------------------------------------------------------- H4: blank over-biased (conditional on the state)
    for rep in range(2):
        if not (deletion_share(best) >= 0.6 and (lr(best) is not None and lr(best) < 0.97)):
            s.note(f"H4 not triggered (deletion share {deletion_share(best):.2f}, length ratio {lr(best)})")
            break
        delta = np.zeros_like(T[OB])
        delta[0] = -0.6
        m4, k4 = probe(
            f"blank_bias_-0.6_{rep + 1}",
            "H4: output still short and deletion-dominated -> the CTC blank bias (+1.23) is too high for this state.",
            "CER drop 0.002-0.01, length ratio toward 1; otherwise blank calibration is fine",
            "add", "add", [OB], "lower the blank bias by 0.6", delta=delta)
        if not k4:
            break
        best = m4

    # ---------------------------------------------------------------- H5: C_3.co.bias (expected refusal)
    m5, k5 = probe(
        "C3_bias_parent",
        "H5: C_3.co.bias has rel L2 1.09 / cos 0.68 to the parent, the largest relative change of any tensor; "
        "but it is tiny (rms 0.018) and its absolute drift matches C_6/C_9, so this is most likely natural.",
        "|dCER| < 0.003 -> refused by the rule; a refusal confirms natural drift, a keep would reveal a planted bias",
        "transplant", "transplant", ["C_3.co.bias"], "parent C_3.co.bias as donor", donor=PT["C_3.co.bias"].copy())
    if k5:
        best = m5

    # ---------------------------------------------------------------- H6: refinement of the L_16 factor (only if H1 kept)
    if k1:
        m6, k6 = probe(
            "L16_x1.1",
            "H6: drift-corrected estimates of the plant factor were 0.47-0.50; test whether a total factor 2.2 beats 2.0.",
            "|dCER| <= 0.005, refused -> factor 2.0 (plant x0.5) is right; kept -> the plant was ~x0.45",
            "scale", "scale", L16, "refine the L_16 factor to 2.2", factor=1.1)
        if k6:
            best = m6

    s.note(f"final state: {describe('best', best)}; passes left {s.budget_left}")


try:
    main()
except Exception as e:  # never leave an undecided edit or an unfrozen session
    s.note(f"exception: {e!r}")
    try:
        s.revert(f"exception: {e!r}")
    except Exception:
        pass
finally:
    s.freeze()
