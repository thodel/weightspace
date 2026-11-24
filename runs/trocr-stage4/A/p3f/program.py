"""TrOCR stage 4, subject A, task (a), budget 10, WITHOUT the donor (p3f). Programmer: claude-fable-5-2.

Written in full before the first pass. No donor is read, loaded or named. See hypotheses.md.

H1  add to dec.11.norm_ffn.bias a direction that raises the logits of word-initial (U+0120) tokens (the missing-space prior)
H2  add to the same tensor a direction that lowers the EOS logit
H3  scale dec.11.norm_ffn.{weight,bias} (logit temperature)
H5  sham for H1 (random vector, same norm, same tensor), always reverted
H4  scale all dec.NN.cross.o by 1.1
"""
import traceback

import numpy as np

from wsp.run import open_session

MAX_PASSES = 10
BIAS = "dec.11.norm_ffn.bias"
GAIN = "dec.11.norm_ffn.weight"
EMB = "dec.embed.tokens"
SP = "Ġ"

s = open_session(".")
MASK = s.unseen_by()                      # no donor: all lines
state = {"passes": 0, "pending": False}


def left() -> int:
    return min(s.budget_left, MAX_PASSES - state["passes"])


def measure(label):
    if left() <= 0:
        raise RuntimeError("pass guard: no pass left")
    m = s.measure(label, rule_mask=MASK)
    state["passes"] += 1
    try:
        s.note(f"{label}: CER {m.cer:.4f} WER {m.wer:.4f} collapsed {m.collapsed}; confusions {str(m.confusions)[:700]}")
    except Exception:
        pass
    return m


def edit(op, names, h, **kw):
    s.edit(op, names, hypothesis=h, **kw)
    state["pending"] = True


def keep(reason) -> bool:
    k = bool(s.keep(reason))              # a refused keep has already undone the edit
    state["pending"] = False
    return k


def revert(reason):
    if state["pending"]:
        s.revert(reason)
    state["pending"] = False


def collapsed(m) -> int:
    try:
        return int(m.collapsed)
    except Exception:
        return 0


def fit_direction(E, target, w, lam_rel=1e-2):
    """Weighted ridge least squares with a free intercept: delta with E[i]@delta ~ target[i] + const.
    A constant shift of all logits is invisible to the softmax, so the intercept costs nothing."""
    idx = np.nonzero(w > 0)[0]
    ww = (w[idx] / w[idx].sum()).astype(np.float64)
    t = target[idx].astype(np.float64)
    d = E.shape[1]
    mu = np.zeros(d)
    for a in range(0, len(idx), 4096):
        j = idx[a:a + 4096]
        mu += (E[j].astype(np.float64) * ww[a:a + 4096, None]).sum(0)
    tbar = float((ww * t).sum())
    G = np.zeros((d, d))
    r = np.zeros(d)
    for a in range(0, len(idx), 4096):
        j = idx[a:a + 4096]
        X = E[j].astype(np.float64) - mu
        Xw = X * ww[a:a + 4096, None]
        G += Xw.T @ X
        r += Xw.T @ (t[a:a + 4096] - tbar)
    lam = lam_rel * float(np.trace(G)) / d
    return np.linalg.solve(G + lam * np.eye(d), r)


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def wstd(x, w):
    m = wmean(x, w)
    return float(np.sqrt(((x - m) ** 2 * w).sum() / w.sum()))


def main():
    T = s.subject.tensors()
    E = T[EMB]
    V = E.shape[0]

    # ---- pass 1: baseline ------------------------------------------------------------------------------------------
    base = measure("C0")
    ref_cer = base.cer                    # CER of the last kept state
    base_col = collapsed(base)

    def sane(m) -> bool:
        return collapsed(m) <= base_col + 10

    # ---- token classes from the model's own tokenizer --------------------------------------------------------------
    tok = s.subject.tokenizer
    strings = list(tok.convert_ids_to_tokens(list(range(V))))
    special = set(int(i) for i in (tok.all_special_ids or []))
    eos = tok.eos_token_id
    cfg_eos = getattr(s.subject.model.config.decoder, "eos_token_id", None)
    if isinstance(cfg_eos, int):
        eos = cfg_eos
    eos = int(eos)
    is_sp = np.array([isinstance(x, str) and x.startswith(SP) for x in strings], dtype=bool)
    w = np.ones(V, dtype=np.float64)

    def short_plain(x):
        if not isinstance(x, str):
            return False
        y = x[1:] if x.startswith(SP) else x
        return 0 < len(y) <= 3 and all(ord(c) < 128 and (c.isalpha() or c in ".,;:-") for c in y)

    for i, x in enumerate(strings):       # short pieces carry a hand the vocabulary was not made for
        if short_plain(x):
            w[i] = 4.0
    for i in special:
        if 0 <= i < V:
            w[i] = 0.0
            is_sp[i] = False
    w_eos = w.copy()
    w[eos] = 200.0                        # H1: EOS pinned to the class of the other tokens
    s.note(f"token classes: vocabulary {V}, word-initial {int(is_sp.sum())}, special {sorted(special)[:8]}, eos id {eos}, "
           f"weight-4 short pieces {int((w == 4.0).sum())}")

    # ---- H1 direction ------------------------------------------------------------------------------------------------
    d1 = fit_direction(E, is_sp.astype(np.float64), w)
    sh = (E.astype(np.float64) @ d1)
    a, b = is_sp & (w > 0), (~is_sp) & (w > 0)
    diff = wmean(sh[a], w[a]) - wmean(sh[b], w[b])
    h1_possible = diff > 1e-8
    if h1_possible:
        d1 = d1 / diff                    # unit: +1 nat for the word-initial class mean against the other class mean
        sh = sh / diff
        sd_a, sd_b = wstd(sh[a], w[a]), wstd(sh[b], w[b])
        s.note(f"H1 direction per nat: norm {np.linalg.norm(d1):.4f} (bias norm {np.linalg.norm(T[BIAS]):.4f}); realised shift "
               f"word-initial mean {wmean(sh[a], w[a]):.3f} sd {sd_a:.3f}, other mean {wmean(sh[b], w[b]):.3f} sd {sd_b:.3f}, "
               f"separation {1.0 / max(sd_a, sd_b, 1e-9):.2f} sd, EOS {sh[eos]:.3f}")
    else:
        s.note("H1: no direction separates word-initial tokens in the embedding (class difference not positive); H1 not run")

    # ---- H1 dose search (each dose from the baseline state; the best one is kept through the rule) ------------------
    kept_h1, kept_delta = False, None
    if h1_possible:
        tried = {}                        # dose -> (cer, sane)

        def try_dose(c):
            h = s.hypothesis(
                f"H1: the readout under-weights word-initial tokens (space -> nothing is 13 % of all edits); add {c} nat to their logits "
                f"through {BIAS} (direction fitted on the model's own embedding)",
                "CER -0.7 points at the best dose (range -0.3 to -1.5); space deletions down by a third to a half",
                "add (readout bias direction, no donor)")
            edit("add", [BIAS], h, delta=(c * d1).astype(np.float32))
            m = measure(f"H1 word-initial +{c} nat")
            tried[c] = (m.cer, sane(m))
            return m

        def good(c):
            return tried[c][1] and tried[c][0] < base.cer

        pending = 1.0
        try_dose(1.0)
        if good(1.0):
            revert("dose search: 1.0 measured, try 2.0")
            pending = 2.0
            try_dose(2.0)
            if good(2.0) and tried[2.0][0] < tried[1.0][0]:
                revert("dose search: 2.0 better than 1.0, try 4.0")
                pending = 4.0
                try_dose(4.0)
        else:
            revert("dose 1.0 does not lower CER, try 0.5")
            pending = 0.5
            try_dose(0.5)
        cands = [c for c in tried if good(c)]
        best = min(cands, key=lambda c: tried[c][0]) if cands else None
        s.note(f"H1 doses (dose: CER, sane): {tried}; baseline {base.cer:.4f}; best {best}")
        if best is None:
            revert("H1: no dose lowers CER on practice")
        elif best == pending:
            kept_h1 = keep(f"H1 dose {best}: lowest CER of the doses tried")
        else:
            revert(f"dose {pending} is not the best; best is {best}")
            if base.cer - tried[best][0] > 0.005 and left() >= 1:
                h = s.hypothesis(f"H1 at its best measured dose {best} (re-applied to be decided by the rule)",
                                 f"CER {tried[best][0]:.4f} as measured before (deterministic pass)", "add (readout bias direction, no donor)")
                edit("add", [BIAS], h, delta=(best * d1).astype(np.float32))
                m = measure(f"H1 word-initial +{best} nat (re-applied)")
                kept_h1 = keep(f"H1 dose {best}: lowest CER of the doses tried")
            else:
                s.note(f"H1: best dose {best} gains {100 * (base.cer - tried[best][0]):.2f} points, below what the keep rule can pass; not re-applied")
        if kept_h1:
            ref_cer = tried[best][0]
            kept_delta = (best * d1).astype(np.float32)

    # ---- H2: EOS logit down ------------------------------------------------------------------------------------------
    if left() >= 1:
        t = np.zeros(V)
        t[eos] = 1.0
        w2 = w_eos.copy()
        w2[eos] = 0.02 * w2.sum()         # the one row that must be hit
        d2 = fit_direction(E, t, w2)
        sh2 = E.astype(np.float64) @ d2
        others = (w_eos > 0)
        others[eos] = False
        gap = sh2[eos] - wmean(sh2[others], w_eos[others])
        if gap > 1e-8:
            d2 = d2 / gap
            sh2 = sh2 / gap
            s.note(f"H2 direction per nat: norm {np.linalg.norm(d2):.4f}; other tokens sd {wstd(sh2[others], w_eos[others]):.3f}")
            h = s.hypothesis("H2: some lines stop early (length ratio 0.952, letter deletions); lower the EOS logit by 1.0 nat through " + BIAS,
                             "CER -0.1 points (range -0.4 to +0.2); collapsed lines up by at most 5; probably not kept",
                             "add (readout bias direction, no donor)")
            edit("add", [BIAS], h, delta=(-1.0 * d2).astype(np.float32))
            m = measure("H2 EOS -1.0 nat")
            if m.cer < ref_cer and sane(m):
                if keep("H2: CER lower than the last kept state"):
                    ref_cer = m.cer
            else:
                revert("H2: CER not lower (or more collapsed lines)")
        else:
            s.note("H2: no direction isolates the EOS row; not run")

    # ---- H3: logit temperature ---------------------------------------------------------------------------------------
    h3_first_worse = False
    if left() >= 1:
        h = s.hypothesis(f"H3: sharper logits (scale {GAIN} and {BIAS} by 1.25, temperature 0.8) change only what beam search does with length",
                         "|dCER| < 0.2 points; not kept", "scale (readout temperature)")
        edit("scale", [GAIN, BIAS], h, factor=1.25)
        m = measure("H3 readout x1.25")
        if m.cer < ref_cer and sane(m):
            if keep("H3: CER lower than the last kept state"):
                ref_cer = m.cer
        else:
            h3_first_worse = m.cer > ref_cer
            revert("H3: CER not lower")

    # ---- H5: sham for H1 (only if H1 was kept) ----------------------------------------------------------------------
    if kept_h1 and kept_delta is not None and left() >= 1:
        rng = np.random.default_rng(20261120)
        r = rng.standard_normal(kept_delta.shape[0])
        r = (r / np.linalg.norm(r) * float(np.linalg.norm(kept_delta))).astype(np.float32)
        h = s.hypothesis("H5 sham: a random vector with the norm of the kept H1 delta on the same tensor does nothing (the H1 effect is its direction)",
                         "|dCER| < 0.15 points; reverted whatever it gives", "add (sham, random direction)")
        edit("add", [BIAS], h, delta=r)
        m = measure("H5 sham random direction")
        s.note(f"H5 sham: CER {m.cer:.4f} against the kept state {ref_cer:.4f}")
        revert("sham: never kept")

    # ---- H3b: softer logits, only if sharper ones hurt ---------------------------------------------------------------
    if h3_first_worse and left() >= 2:
        h = s.hypothesis(f"H3b: softer logits (scale {GAIN} and {BIAS} by 0.8) since x1.25 raised CER",
                         "CER -0.1 points (range -0.3 to +0.2); not kept", "scale (readout temperature)")
        edit("scale", [GAIN, BIAS], h, factor=0.8)
        m = measure("H3b readout x0.8")
        if m.cer < ref_cer and sane(m):
            if keep("H3b: CER lower than the last kept state"):
                ref_cer = m.cer
        else:
            revert("H3b: CER not lower")

    # ---- H4: cross-attention output gain ----------------------------------------------------------------------------
    if left() >= 1:
        names = [f"dec.{i:02d}.cross.o.{p}" for i in range(12) for p in ("weight", "bias")]
        h = s.hypothesis("H4: more weight on the image against the decoder's own context: all dec.NN.cross.o x1.1",
                         "CER between -0.1 and +0.3 points; not kept; a global scale is what a random comparator also gives",
                         "scale (cross-attention gain, all decoder layers)")
        edit("scale", names, h, factor=1.1)
        m = measure("H4 cross.o x1.1")
        if m.cer < ref_cer and sane(m):
            if keep("H4: CER lower than the last kept state"):
                ref_cer = m.cer
        else:
            revert("H4: CER not lower")

    s.note(f"end: baseline CER {base.cer:.4f}, last kept state CER {ref_cer:.4f}, H1 kept {kept_h1}, passes used {state['passes']}")


try:
    main()
except Exception:
    try:
        s.note("program error, pending edit reverted: " + traceback.format_exc()[-1500:])
    except Exception:
        pass
    try:
        revert("program error")
    except Exception:
        pass
finally:
    if state["pending"]:
        try:
            s.revert("end of program with an edit pending")
        except Exception:
            pass
    summary = s.freeze()
    print({k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})
