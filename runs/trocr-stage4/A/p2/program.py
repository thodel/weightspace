"""Stage 4, subject A, task (a), budget 10, WITH the donor. Programmer p2 (claude-fable-5-2).

Written in full before the first pass; see hypotheses.md. At most 10 measure calls:
  1 C0 | 2-4 H1 space offset ladder | 5 H2 end-token offset | 6-7 H3 extrapolation away from the donor |
  8 H4 encoder toward the donor (contrast) | 9 H5 combination (conditional). One pass stays in reserve.
"""
from collections import Counter

import numpy as np

from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

DONOR = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
BIAS = "dec.11.norm_ffn.bias"
EMB = "dec.embed.tokens"
SP = "Ġ"

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
state = {"cer": None, "collapsed": None, "cer0": None}


def note(text):
    print(text, flush=True)
    s.note(text)


def conf_text(m):
    try:
        return str(getattr(m, "confusions", None))[:700]
    except Exception:
        return "(confusions not readable)"


def trial(label, text, prediction, eclass, op, names, **kw):
    """hypothesis -> edit -> one pass -> keep (the rule decides) or revert. Returns (kept, metrics, delta) or None."""
    if s.budget_left < 1:
        note(f"{label}: no pass left, not run")
        return None
    h = s.hypothesis(text, prediction, eclass)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=MASK)
    d = m.cer - state["cer"]
    if m.collapsed > state["collapsed"] + 10:
        s.revert(f"{label}: collapsed lines {state['collapsed']} -> {m.collapsed}")
        kept = False
    elif d >= 0:
        s.revert(f"{label}: CER not lower ({state['cer']:.4f} -> {m.cer:.4f})")
        kept = False
    else:
        kept = bool(s.keep(f"{label}: CER {state['cer']:.4f} -> {m.cer:.4f}"))
    note(f"{label}: CER {m.cer:.4f} (delta {d:+.4f} vs kept state), WER {m.wer:.4f}, collapsed {m.collapsed}, kept {kept}; "
         f"confusions {conf_text(m)}")
    if kept:
        state["cer"], state["collapsed"] = m.cer, m.collapsed
    return kept, m, d


# ---------------------------------------------------------------------------------------------------------------------
# the logit-offset vector: weighted ridge so that E @ d is the target per token
# ---------------------------------------------------------------------------------------------------------------------
def solve_offset(E, target, weight):
    V, H = E.shape
    G = np.zeros((H, H), dtype=np.float64)
    r = np.zeros(H, dtype=np.float64)
    for i in range(0, V, 8192):
        Ec = E[i:i + 8192].astype(np.float64)
        wc = weight[i:i + 8192]
        G += (Ec * wc[:, None]).T @ Ec
        r += Ec.T @ (wc * target[i:i + 8192])
    lam = 1e-3 * np.trace(G) / H
    return np.linalg.solve(G + lam * np.eye(H), r)


def shifts_of(E, d):
    out = np.empty(E.shape[0], dtype=np.float64)
    for i in range(0, E.shape[0], 8192):
        out[i:i + 8192] = E[i:i + 8192].astype(np.float64) @ d
    return out


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def wstd(x, w):
    mu = wmean(x, w)
    return float(np.sqrt((w * (x - mu) ** 2).sum() / w.sum()))


def build_vocab():
    tok = s.subject.tokenizer
    E = s.subject.tensors()[EMB]
    V = E.shape[0]
    strings = tok.convert_ids_to_tokens(list(range(V)))
    is_sp = np.array([isinstance(t, str) and t.startswith(SP) for t in strings], dtype=bool)
    special = sorted({i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id) if i is not None and 0 <= i < V})
    cnt = Counter()
    for it in s.practice:
        ids = tok(it["text"]).input_ids
        cnt.update(i for i in ids if i not in special)
    freq = np.zeros(V, dtype=np.float64)
    for i, c in cnt.items():
        freq[i] = c
    eos = tok.eos_token_id
    return E, is_sp, special, freq, eos


def space_direction(E, is_sp, special, freq):
    """Unit-gap vector: frequency-weighted mean shift of used word-initial tokens minus that of used other tokens = 1."""
    used = freq > 0
    target = is_sp.astype(np.float64)
    weight = np.where(used, 100.0, 1.0)
    for i in special:
        target[i] = 0.0
        weight[i] = 5000.0
    d = solve_offset(E, target, weight)
    sh = shifts_of(E, d)
    g, n = used & is_sp, used & ~is_sp
    gap = wmean(sh[g], freq[g]) - wmean(sh[n], freq[n])
    if not np.isfinite(gap) or gap <= 0.05:
        return None, f"space direction: gap {gap} on the used tokens, no usable direction"
    d = d / gap
    sh = sh / gap
    centre = wmean(sh[n], freq[n])
    info = (f"space direction (per unit b): used word-initial types {int(g.sum())}, used other types {int(n.sum())}; "
            f"shift of used other tokens mean {centre:+.3f} sd {wstd(sh[n], freq[n]):.3f}; of used word-initial mean "
            f"{wmean(sh[g], freq[g]):+.3f} sd {wstd(sh[g], freq[g]):.3f}; specials {[round(float(sh[i]), 3) for i in special]}; "
            f"largest shift of an unused non-initial token {float(sh[~used & ~is_sp].max()):+.3f}; |d| {float(np.linalg.norm(d)):.4f}")
    return d, info


def eos_direction(E, special, freq, eos):
    """Unit vector in logit terms: shift of the end token minus the mean shift of the used tokens = -1."""
    used = freq > 0
    target = np.zeros(E.shape[0], dtype=np.float64)
    weight = np.where(used, 100.0, 1.0)
    for i in special:
        weight[i] = 100.0
    target[eos] = -1.0
    weight[eos] = 20000.0
    d = solve_offset(E, target, weight)
    sh = shifts_of(E, d)
    gap = float(sh[eos]) - wmean(sh[used], freq[used])
    if not np.isfinite(gap) or gap >= -0.05:
        return None, f"end-token direction: gap {gap}, no usable direction"
    d = d / (-gap)
    sh = sh / (-gap)
    info = (f"end-token direction (per unit c): end token {float(sh[eos]):+.3f}, used tokens mean {wmean(sh[used], freq[used]):+.3f} "
            f"sd {wstd(sh[used], freq[used]):.3f}, |d| {float(np.linalg.norm(d)):.4f}")
    return d, info


def main():
    base = s.measure("C0", rule_mask=MASK)
    state["cer"], state["collapsed"], state["cer0"] = base.cer, base.collapsed, base.cer
    note(f"C0: CER {base.cer:.4f}, WER {base.wer:.4f}, collapsed {base.collapsed}; rule lines {sum(MASK)} of {len(MASK)}; confusions {conf_text(base)}")

    E, is_sp, special, freq, eos = build_vocab()
    best_s = None        # (delta vs C0, b) of the best H1 pass that was not kept
    s_kept = False
    eos_kept = False

    # ---- H1: space offset --------------------------------------------------------------------------------------------
    d_sp, info = space_direction(E, is_sp, special, freq)
    note(info)

    def space_trial(b, tag):
        return trial(f"H1 space offset b={b} ({tag})",
                     "The model under-writes word-initial tokens (1,540 missing spaces, hardly any inserted): a logit offset of +b on "
                     "tokens with a leading U+0120, written into dec.11.norm_ffn.bias in front of the tied readout, restores part of them.",
                     "CER down 0.005 to 0.015 at the best b; space deletions down by a third or more",
                     "logit offset through the final layer-norm bias",
                     "add", [BIAS], delta=(b * d_sp).astype(np.float32))

    if d_sp is not None:
        r = space_trial(1.0, "first")
        if r is not None:
            kept, m, d = r
            if kept:
                s_kept = True
                space_trial(1.0, "second step on top of the kept 1.0")
            elif d < -0.001:
                best_s = (d, 1.0)
                r2 = space_trial(2.0, "from the unedited bias")
                if r2 is not None:
                    k2, m2, d2 = r2
                    if k2:
                        s_kept = True
                    elif d2 < best_s[0]:
                        best_s = (d2, 2.0)
                        r3 = space_trial(3.0, "from the unedited bias")
                        if r3 is not None:
                            k3, m3, d3 = r3
                            if k3:
                                s_kept = True
                            elif d3 < best_s[0]:
                                best_s = (d3, 3.0)
            else:
                r2 = space_trial(0.5, "smaller, after 1.0 did not lower CER")
                if r2 is not None:
                    k2, m2, d2 = r2
                    if k2:
                        s_kept = True
                    elif d2 < -0.001:
                        best_s = (d2, 0.5)

    # ---- H2: end-token offset ----------------------------------------------------------------------------------------
    d_eos, info = eos_direction(E, special, freq, eos)
    note(info)
    if d_eos is not None:
        r = trial("H2 end-token offset c=1.0",
                  "Length ratio 0.952 and 29 collapsed lines: the end token is written too early; -1.0 on its logit through dec.11.norm_ffn.bias.",
                  "CER down 0.001 to 0.003, collapsed lines down by a few; not expected to pass the rule",
                  "logit offset through the final layer-norm bias",
                  "add", [BIAS], delta=(1.0 * d_eos).astype(np.float32))
        eos_kept = bool(r and r[0])

    # ---- donor ---------------------------------------------------------------------------------------------------------
    D = TrocrSubject(DONOR, device="cpu", check_card=False).tensors()
    T = s.subject.tensors()
    common = sorted(n for n in T if n in D and D[n].shape == T[n].shape)
    note(f"donor loaded: {len(common)} of {len(T)} tensors have a donor tensor of the same shape")
    x_names = [n for n in common if n != EMB]
    enc_names = [n for n in common if n.startswith("enc.")]
    # a reading from the weights themselves (not from the pack): where the two models put word-initial tokens at the readout
    try:
        used = freq > 0
        g, n_ = used & is_sp, used & ~is_sp
        for who, tens in (("subject", s.subject.tensors()), ("donor", D)):
            sh = shifts_of(tens[EMB], tens[BIAS].astype(np.float64))
            note(f"readout of the final-norm bias, {who}: used word-initial {wmean(sh[g], freq[g]):+.3f}, used other {wmean(sh[n_], freq[n_]):+.3f}, "
                 f"end token {float(sh[eos]):+.3f}")
    except Exception as exc:          # a note only; never decides anything
        note(f"bias readout note failed: {exc!r}")

    best_x = None        # (delta vs kept state at the time, a) of the best H3 pass that was not kept
    x_kept = False
    cer_before_x = state["cer"]

    def extra_trial(a, tag):
        return trial(f"H3 extrapolation away from the donor a={a} ({tag})",
                     "The fine-tuning step is small and the model still reads its training hand at 23 %: lengthening the step, "
                     "w + a (w - donor) on every tensor but the tied embedding, lowers CER on this hand.",
                     "CER down 0.002 to 0.008 (low confidence, about one in three that it is lower at all)",
                     "extrapolation away from the donor (interpolate, negative alpha), all tensors but dec.embed.tokens",
                     "interpolate", x_names, alpha=-float(a), per_tensor={n: {"other": D[n]} for n in x_names})

    r = extra_trial(0.3, "first")
    if r is not None:
        kept, m, d = r
        if kept:
            x_kept = True
            extra_trial(0.3, "second step on top of the kept one")
        elif d < -0.001:
            best_x = (d, 0.3)
            r2 = extra_trial(0.6, "from the kept state")
            if r2 is not None:
                k2, m2, d2 = r2
                if k2:
                    x_kept = True
                elif d2 < best_x[0]:
                    best_x = (d2, 0.6)
        else:
            r2 = extra_trial(0.15, "smaller, after 0.3 did not lower CER")
            if r2 is not None:
                k2, m2, d2 = r2
                if k2:
                    x_kept = True
                elif d2 < -0.001:
                    best_x = (d2, 0.15)

    # ---- H4: encoder toward the donor (contrast) ---------------------------------------------------------------------
    if s.budget_left >= 2 or (s.budget_left >= 1 and (best_s is None or best_x is None or s_kept or x_kept)):
        trial("H4 encoder toward the donor alpha=0.3",
              "Contrast for H3: pulling the encoder back toward the donor does not help on this hand.",
              "CER unchanged or up by 0 to 0.005; reverted",
              "partial interpolation toward the donor (encoder tensors only)",
              "interpolate", enc_names, alpha=0.3, per_tensor={n: {"other": D[n]} for n in enc_names})

    # ---- H5: combination, only if both lowered CER by more than 0.002 and nothing of either family was kept ----------------
    if (not s_kept and not x_kept and not eos_kept and best_s is not None and best_x is not None
            and best_s[0] < -0.002 and best_x[0] < -0.002 and d_sp is not None and s.budget_left >= 1):
        b, a = best_s[1], best_x[1]
        Tc = s.subject.tensors()
        per = {}
        for n in x_names:
            delta = (a * (Tc[n].astype(np.float64) - D[n].astype(np.float64)))
            if n == BIAS:
                delta = delta + b * d_sp
            per[n] = {"delta": delta.astype(np.float32)}
        trial(f"H5 space offset b={b} with extrapolation a={a}",
              "The readout offset and the longer step act on different places and their gains add.",
              "CER down 0.005 to 0.015, enough for the keep rule where each alone was not",
              "one add over all tensors but dec.embed.tokens: extrapolation away from the donor plus the space offset",
              "add", x_names, per_tensor=per)
    else:
        note(f"H5 not run: s_kept {s_kept}, x_kept {x_kept}, eos_kept {eos_kept}, best_s {best_s}, best_x {best_x}, passes left {s.budget_left}")

    note(f"end: CER of the kept state {state['cer']:.4f} against C0 {state['cer0']:.4f}; passes left {s.budget_left}; "
         f"H1 kept {s_kept}, H2 kept {eos_kept}, H3 kept {x_kept} (H3 started from {cer_before_x:.4f})")


try:
    main()
except Exception as exc:                      # leave a decided, frozen session behind whatever happens
    import traceback
    traceback.print_exc()
    try:
        s.revert(f"program error: {exc!r}")   # a logged refusal if no edit is pending
    except Exception:
        pass
    try:
        s.note(f"program stopped on an error: {exc!r}")
    except Exception:
        pass
finally:
    summary = s.freeze()
    print({k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")}, flush=True)
