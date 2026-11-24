"""Stage 4, subject B, task (a), budget 10, no donor. Global one-scalar knobs, each measured and decided by the keep rule.
See hypotheses.md. At most 10 measure calls (1 baseline + 9 trials)."""
from wsp.run import open_session
import numpy as np

s = open_session(".")
MASK = s.unseen_by()                       # no donor: all lines
L = range(12)


def names(part):
    return [f"dec.{i:02d}.{part}.{p}" for i in L for p in ("weight", "bias")]


CROSS_O, SELF_O, CROSS_Q = names("cross.o"), names("self.o"), names("cross.q")
LAST_NORM = ["dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"]
BIAS = "dec.11.norm_ffn.bias"

state = {"cer": None, "collapsed": None}


def left():
    return s.budget_left > 0


m0 = s.measure("C0", rule_mask=MASK)
state["cer"], state["collapsed"] = m0.cer, m0.collapsed
s.note(f"C0 cer={m0.cer:.4f} wer={m0.wer:.4f} collapsed={m0.collapsed}")


def trial(label, text, prediction, cls, op, tensor_names, **kwargs):
    """One hypothesis, one edit, one pass, one decision. Returns (kept, cer) or None when no pass is left."""
    if not left():
        s.note(f"{label}: skipped, no pass left")
        return None
    h = s.hypothesis(text, prediction, cls)
    s.edit(op, tensor_names, hypothesis=h, **kwargs)
    m = s.measure(label, rule_mask=MASK)
    d = m.cer - state["cer"]
    if d < 0 and m.collapsed <= state["collapsed"] + 5:
        kept = s.keep(f"{label}: CER {d:+.4f} against the last kept state; the rule decides")
    else:
        s.revert(f"{label}: CER {d:+.4f}, collapsed {state['collapsed']} -> {m.collapsed}; not better")
        kept = False
    if kept:
        state["cer"], state["collapsed"] = m.cer, m.collapsed
    s.note(f"{label}: cer={m.cer:.4f} delta={d:+.4f} collapsed={m.collapsed} kept={kept}")
    return kept, d


# ---- bias directions on the readout (logits = E @ (g*z + b); a change of b moves logits by E @ delta) -------------------
def bias_directions():
    """EOS direction and word-initial direction, relative to the tokens that occur in the practice references (the
    vocabulary of the hand; no fitting to errors). Returns (delta_eos_per_nat, delta_space_per_nat) or (None, None)."""
    try:
        tok = s.subject.tokenizer
        E = s.subject.tensors()["dec.embed.tokens"].astype(np.float64)
        ids = set()
        for it in s.practice:
            ids.update(tok(it["text"], add_special_tokens=False).input_ids)
        ids = sorted(i for i in ids if 0 <= i < E.shape[0])
        if len(ids) < 20:
            return None, None
        strs = tok.convert_ids_to_tokens(ids)
        sp = [i for i, t in zip(ids, strs) if t.startswith("Ġ")]
        ns = [i for i, t in zip(ids, strs) if not t.startswith("Ġ")]
        eos = tok.eos_token_id
        d_eos = E[eos] - E[ids].mean(axis=0)
        de = d_eos / float(d_eos @ d_eos)                 # +1 nat on EOS against the mean active token
        dsp = None
        if len(sp) >= 5 and len(ns) >= 5:
            d = E[sp].mean(axis=0) - E[ns].mean(axis=0)
            dsp = d / float(d @ d)                        # +1 nat on word-initial against the other active tokens (means)
        return de, dsp
    except Exception as exc:                              # never lose the session to a helper
        s.note(f"bias directions unavailable: {exc!r}")
        return None, None


DE, DSP = bias_directions()

# ---- H1 / H1b: image share (cross-attention write) --------------------------------------------------------------------
r1 = trial("H1 cross.o x1.15", "A larger cross-attention write raises the image share against the language prior and cuts deletions.",
           "CER -0.2 points (range -1.0..+0.5); not kept", "scale dec cross.o all layers", "scale", CROSS_O, factor=1.15)
if r1 is not None:
    f = 1.15 if r1[1] < 0 else 0.87
    trial(f"H1b cross.o x{f}", "Follow-up of H1: stronger if CER fell, opposite sign if it rose.",
          "within +-0.5 points; not kept", "scale dec cross.o all layers", "scale", CROSS_O, factor=f)

# ---- H2..H6: knobs with a signed strength t (t=1 nominal, 2 doubled, -1 opposite) --------------------------------------
KNOBS = {}


def knob_scale(tensor_names, base):
    return lambda t: ("scale", tensor_names, {"factor": float(base ** t)})


def knob_add(vec, size):
    return lambda t: ("add", [BIAS], {"delta": (vec * size * t).astype(np.float32)})


KNOBS["H2 self.o x0.90"] = (knob_scale(SELF_O, 0.90), "A smaller self-attention write weakens the language prior.",
                            "CER 0..+0.3 points; not kept", "scale dec self.o all layers")
KNOBS["H3 cross.q x1.15"] = (knob_scale(CROSS_Q, 1.15), "Sharper cross-attention tracks the line better: fewer skipped characters and loops.",
                             "CER -0.1 points (range -0.7..+0.5); not kept", "scale dec cross.q all layers")
if DE is not None:
    KNOBS["H4 EOS -0.7 nat"] = (knob_add(DE, -0.7), "The end token wins too early (length ratio 0.958): lower its logit by 0.7 nat via the last norm's bias.",
                                "CER -0.1 points (range -0.5..+0.3); not kept", "add dec.11.norm_ffn.bias one direction")
if DSP is not None:
    KNOBS["H5 word-initial +0.3 nat"] = (knob_add(DSP, 0.3), "Spaces are deleted 660x, inserted 300x: shift word-initial tokens by +0.3 nat via the last norm's bias.",
                                         "CER -0.2 points (range -0.6..+0.4); not kept", "add dec.11.norm_ffn.bias one direction")
KNOBS["H6 logit temperature x1.25"] = (knob_scale(LAST_NORM, 1.25), "A sharper readout changes the beam ranking between short and long hypotheses.",
                                       "CER 0 +-0.2 points; not kept", "scale dec.11.norm_ffn weight+bias")

results = {}
for label, (make, text, pred, cls) in KNOBS.items():
    if s.budget_left <= 2:                               # two passes are held back for the follow-ups
        s.note(f"{label}: not run, passes held for follow-ups")
        continue
    op, tn, kw = make(1.0)
    r = trial(label, text, pred, cls, op, tn, **kw)
    if r is not None:
        results[label] = r

# ---- F1, F2: follow-ups on the two best not-kept knobs ------------------------------------------------------------------
cands = sorted(((d, label) for label, (kept, d) in results.items() if not kept), key=lambda x: x[0])
done = 0
for d, label in cands:
    if done >= 2 or not left():
        break
    make, text, pred, cls = KNOBS[label]
    t = 2.0 if d < 0 else -1.0
    op, tn, kw = make(t)
    trial(f"F{done + 1} {label} t={t:+.0f}", f"Follow-up of '{label}' (first delta {d:+.4f}): "
          + ("double the step." if d < 0 else "opposite sign."),
          "at most -0.5 points; probably not kept", cls, op, tn, **kw)
    done += 1

s.note(f"end: kept-state cer={state['cer']:.4f} (C0 {m0.cer:.4f}), passes left {s.budget_left}")
summary = s.freeze()
