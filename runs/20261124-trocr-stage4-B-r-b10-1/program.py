"""TrOCR stage 4, subject B, arm R: budget 10, with the donor. Written in full before the first pass (see hypotheses.md).

Passes: 1 C0; 2-3 H1 extrapolation away from the donor; 4-5 H2 cross-attention output gain; 6-7 H3 cross-attention
temperature; 8-9 H4 end-token row shift; 10 unspent. Every edit is decided by s.keep (the harness rule) or reverted here.
"""
import sys

import numpy as np

from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

DONOR = "/home/tobias/atr-cache/trocr/microsoft__trocr-base-handwritten"
EMBED = ("dec.embed.tokens", "dec.embed.pos")
MAX_PASSES = 10

s = open_session(".")
MASK = s.unseen_by("trocr-base-handwritten")
state = {"cer": None, "collapsed": None, "passes": 0}


def finish(code=0):
    summary = s.freeze()
    print("frozen:", {k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})
    sys.exit(code)


def describe(m):
    lr = getattr(m, "length_ratio", None)
    return f"CER {m.cer:.4f} WER {getattr(m, 'wer', float('nan')):.4f} collapsed {m.collapsed} length ratio {lr}"


def can_pass():
    return state["passes"] < MAX_PASSES and s.budget_left >= 1


def trial(label, text, prediction, edit_class, op, names, per_tensor=None, **kwargs):
    """hypothesis -> edit -> one pass -> keep (rule) or revert. Returns (kept, CER change against the last kept state),
    or (False, None) when no pass is left. The edit is applied only when a pass is left to decide it."""
    if not can_pass():
        s.note(f"{label}: skipped, no pass left")
        return False, None
    h = s.hypothesis(text, prediction, edit_class)
    if per_tensor is not None:
        s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kwargs)
    else:
        s.edit(op, names, hypothesis=h, **kwargs)
    try:
        m = s.measure(label, rule_mask=MASK)
    except Exception as exc:                      # a refused or failed pass: undo, freeze the state as it stands
        s.revert(f"{label}: pass failed ({type(exc).__name__}: {exc}); edit undone")
        s.note("session ends here: a pass failed, the pending edit was reverted")
        finish(1)
    state["passes"] += 1
    delta = m.cer - state["cer"]
    try:
        conf = str(m.confusions)[:400]
    except Exception:
        conf = "n/a"
    s.note(f"{label}: {describe(m)}; change against the last kept state {delta:+.4f}; confusions {conf}")
    print(label, describe(m), f"delta {delta:+.4f}")
    if delta < 0 and m.collapsed <= state["collapsed"] + 5:
        kept = s.keep(f"{label}: practice CER {delta:+.4f}, collapsed {m.collapsed} (was {state['collapsed']}); the rule decides")
        if kept:
            state["cer"], state["collapsed"] = m.cer, m.collapsed
    else:
        s.revert(f"{label}: practice CER {delta:+.4f}, collapsed {m.collapsed} (was {state['collapsed']}): not better, reverted")
        kept = False
    return kept, delta


# ---- pass 1: C0 --------------------------------------------------------------------------------------------------------
try:
    m0 = s.measure("C0", rule_mask=MASK)
except Exception as exc:
    s.note(f"C0 failed ({type(exc).__name__}: {exc}); nothing edited")
    finish(1)
state.update(cer=m0.cer, collapsed=m0.collapsed, passes=1)
s.note(f"C0: {describe(m0)}; rule lines {sum(MASK)} of {len(MASK)}")
print("C0", describe(m0))

# ---- the donor (arrays only; the donor model is dropped) --------------------------------------------------------------
donor_subject = TrocrSubject(DONOR, device="cpu", check_card=False)
D = {k: v.copy() for k, v in donor_subject.tensors().items()}
del donor_subject
T = s.subject.tensors()
shared = sorted(n for n in T if n in D and D[n].shape == T[n].shape and n not in EMBED)
missing = sorted(n for n in T if n not in D or D[n].shape != T[n].shape)
s.note(f"donor loaded: {len(D)} tensors, {len(shared)} shared with the subject outside the decoder embeddings; not shared: {missing[:10]}")
enc_names = [n for n in shared if n.startswith("enc.")]

# ---- H1: extrapolation away from the donor (interpolate with alpha < 0) -------------------------------------------------
kept1, d1 = trial(
    "H1 extrapolate 0.10 all but embeddings",
    "The fine-tune is under-converged (CER 28 %): a step further along donor->subject, w + 0.10 (w - donor), on every tensor "
    "except dec.embed.tokens and dec.embed.pos, continues the adaptation.",
    "practice CER -0.2 to -0.6 points, more likely inside the noise; not kept by the rule",
    "interpolate (alpha -0.10, extrapolation away from the donor)",
    "interpolate", shared, per_tensor={n: {"other": D[n]} for n in shared}, alpha=-0.10)
if kept1:
    trial(
        "H1b extrapolate a second 0.10 step",
        "H1 was kept: the line donor->subject has more to give; a second step of 0.10 from the kept state.",
        "practice CER another -0.2 points or nothing",
        "interpolate (alpha -0.10, second step)",
        "interpolate", shared, per_tensor={n: {"other": D[n]} for n in shared}, alpha=-0.10)
else:
    trial(
        "H1b extrapolate 0.20 encoder only",
        "The encoder moved least from the donor while the script changed most: w + 0.20 (w - donor) on the enc.* tensors only.",
        "practice CER -0.1 to -0.4 points, probably noise; not kept by the rule",
        "interpolate (alpha -0.20, encoder only)",
        "interpolate", enc_names, per_tensor={n: {"other": D[n]} for n in enc_names}, alpha=-0.20)
del D

# ---- H2: cross-attention output gain -------------------------------------------------------------------------------------
cross_o = [f"dec.{i:02d}.cross.o.{p}" for i in range(12) for p in ("weight", "bias")]
cross_o = [n for n in cross_o if n in s.subject.tensors()]
kept2, d2 = trial(
    "H2 cross.o x1.15",
    "Deletions dominate (space, e, n, r, s, i, c): the decoder under-uses the image against its own context. Raise the gain of "
    "every cross-attention output (dec.NN.cross.o weight and bias, 12 layers) by 1.15.",
    "practice CER -0.2 to -0.5 points with fewer deletions; could be +0.3",
    "scale (cross-attention output, all decoder layers)",
    "scale", cross_o, factor=1.15)
if d2 is not None:
    if d2 < 0:
        f2, why2 = (1.30 if not kept2 else 1.15), "H2 lowered CER: a larger dose from the last kept state"
    else:
        f2, why2 = 0.87, "H2 raised CER: the opposite direction, less cross-attention gain"
    trial(
        f"H2b cross.o x{f2:.2f}",
        why2 + f" (dec.NN.cross.o weight and bias x {f2:.2f}).",
        "practice CER within +-0.3 points; if both directions lose, the gain is at its optimum",
        "scale (cross-attention output, dose step)",
        "scale", cross_o, factor=f2)

# ---- H3: cross-attention temperature ---------------------------------------------------------------------------------------
cross_q = [f"dec.{i:02d}.cross.q.{p}" for i in range(12) for p in ("weight", "bias")]
cross_q = [n for n in cross_q if n in s.subject.tensors()]
kept3, d3 = trial(
    "H3 cross.q x1.15",
    "Skipped glyphs: cross-attention is too diffuse to step glyph by glyph. Sharpen it by a temperature: dec.NN.cross.q weight "
    "and bias x 1.15 in all 12 layers (the attention logits scale by 1.15, nothing else changes).",
    "practice CER -0.1 to -0.4 points, or noise",
    "scale (cross-attention query, all decoder layers)",
    "scale", cross_q, factor=1.15)
if d3 is not None:
    if d3 < 0:
        f3, why3 = (1.30 if not kept3 else 1.15), "H3 lowered CER: a sharper temperature from the last kept state"
    else:
        f3, why3 = 0.87, "H3 raised CER: the opposite direction, a softer cross-attention"
    trial(
        f"H3b cross.q x{f3:.2f}",
        why3 + f" (dec.NN.cross.q weight and bias x {f3:.2f}).",
        "practice CER within +-0.3 points; if both directions lose, the temperature is at its optimum",
        "scale (cross-attention query, dose step)",
        "scale", cross_q, factor=f3)


# ---- H4: end-token row against the constant part of the final decoder state ---------------------------------------------
def eos_row_values(nats):
    """The </s> row moved so that its logit against dec.11.norm_ffn.bias drops by ``nats``; step capped at 0.35 of the row norm."""
    T = s.subject.tensors()
    tok = getattr(s.subject, "tokenizer", None)
    eos = getattr(tok, "eos_token_id", None)
    eos = 2 if eos is None else int(eos)
    b = T["dec.11.norm_ffn.bias"].astype(np.float64)
    row = T["dec.embed.tokens"][eos].astype(np.float64)
    step = -nats * b / float(b @ b)
    cap = 0.35 * float(np.linalg.norm(row))
    n = float(np.linalg.norm(step))
    if n > cap > 0:
        step *= cap / n
    info = (f"eos id {eos}, row norm {np.linalg.norm(row):.3f}, bias norm {np.linalg.norm(b):.3f}, row.bias before {row @ b:+.3f}, "
            f"step norm {np.linalg.norm(step):.3f}, row.bias change {step @ b:+.3f}")
    return eos, (row + step).astype(np.float32)[None, :], info


if can_pass():
    eos, values, info = eos_row_values(1.5)
    s.note("H4 geometry: " + info)
    kept4, d4 = trial(
        "H4 eos row -1.5 nats along the final bias",
        "Part of the missing text is an early end of line: lower the constant part of the end-token logit by moving the </s> row of "
        "the tied embedding against dec.11.norm_ffn.bias (about 1.5 nats, step capped at 0.35 of the row norm). " + info,
        "length ratio up by under 0.01, practice CER -0.1 to -0.3 points; weak (the bias is a small part of the final state, and </s> "
        "is also the decoder start token)",
        "set_rows (one row of dec.embed.tokens)",
        "set_rows", ["dec.embed.tokens"], index=[eos], values=values)
    if d4 is not None and d4 < 0 and can_pass():
        eos, values, info = eos_row_values(1.5)
        trial(
            "H4b eos row a second step",
            "H4 lowered CER: the same step once more from the last kept state. " + info,
            "practice CER another -0.1 points or nothing",
            "set_rows (one row of dec.embed.tokens, second step)",
            "set_rows", ["dec.embed.tokens"], index=[eos], values=values)

s.note(f"end: {state['passes']} passes used by this program, last kept state CER {state['cer']:.4f}, collapsed {state['collapsed']}; "
       f"baseline CER {m0.cer:.4f}")
finish(0)
