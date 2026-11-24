"""Subject B, task (a), budget 10, with the donor (p2). Written in full before the first pass; see hypotheses.md.

At most 10 measure calls: C0, H1 (1-2), H2 (1-2), H3 (1-2), H4 (1), H5 (the rest, at most 2).
"""
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

DONOR = "/home/tobias/atr-cache/trocr/microsoft__trocr-base-handwritten"
STEP = 0.001            # 0.1 CER point: below this a change has no direction for a follow-up
MAX_COLLAPSE_RISE = 5

s = open_session(".")
MASK = s.unseen_by("trocr-base-handwritten")
LAYERS = [f"{i:02d}" for i in range(12)]

state = {"cer": None, "collapsed": None}


def describe(m):
    lr = getattr(m, "length_ratio", None)
    return f"CER {m.cer:.4f}, collapsed {m.collapsed}, length ratio {lr}"


def trial(label, text, prediction, edit_class, op, names, per_tensor=None, **kwargs):
    """hypothesis -> edit -> one pass -> keep (rule) or revert (collapse guard). Returns (delta CER vs last kept, kept) or (None, False)."""
    if s.budget_left < 1:
        s.note(f"{label}: skipped, no pass left")
        return None, False
    h = s.hypothesis(text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kwargs)
    m = s.measure(label, rule_mask=MASK)
    delta = m.cer - state["cer"]
    if m.collapsed > state["collapsed"] + MAX_COLLAPSE_RISE:
        s.revert(f"{label}: collapsed lines {state['collapsed']} -> {m.collapsed}; delta CER {delta:+.4f}")
        kept = False
    else:
        kept = s.keep(f"{label}: delta CER {delta:+.4f} against the last kept state")
    s.note(f"{label}: {describe(m)}; delta {delta:+.4f}; kept {kept}; top confusions {list(m.confusions)[:6] if m.confusions else None}")
    if kept:
        state["cer"], state["collapsed"] = m.cer, m.collapsed
    return delta, kept


def scale_with_followup(tag, names, text, pred, cls, f1, f_more, f_back):
    """First factor f1; then one follow-up: f_more on top if it helped (or was kept), f_back from the original if it hurt."""
    d, kept = trial(f"{tag} x{f1}", text, pred, cls, "scale", names, factor=f1)
    if d is None:
        return
    if kept or d < -STEP:
        # after a keep the edit is in place: step on from there; after a refusal the weights are back: apply the full factor
        f = f_more if kept else f1 * f_more
        trial(f"{tag} x{f1 * f_more:.3f} total", text + " (stronger, same direction)",
              "a further fall of the same size or less; a rise means the first step was already past the optimum", cls,
              "scale", names, factor=f)
    elif d > STEP:
        trial(f"{tag} x{f_back}", text + " (opposite direction, since the first direction raised the CER)",
              "CER -0.1 to -0.3 points", cls, "scale", names, factor=f_back)
    else:
        s.note(f"{tag}: change below 0.1 point, no follow-up")


# ---- pass 1: C0 ------------------------------------------------------------------------------------------------------
m0 = s.measure("C0", rule_mask=MASK)
state["cer"], state["collapsed"] = m0.cer, m0.collapsed
s.note(f"C0: {describe(m0)}; rule lines {sum(MASK)} of {len(MASK)}")

# ---- H1: cross-attention sharpness -----------------------------------------------------------------------------------
names_q = [f"dec.{l}.cross.q.{p}" for l in LAYERS for p in ("weight", "bias")]
scale_with_followup("H1 cross.q", names_q,
                    "Lost characters (e n r s i c, spaces) come from a cross-attention that skips glyphs; scaling the queries of all 12 "
                    "cross-attentions multiplies their attention logits and sharpens the alignment",
                    "CER -0.1 to -0.4 points, fewer deletions, collapsed not up; too small for the keep rule",
                    "scale: decoder cross-attention query, all layers", 1.15, 1.15, 0.87)

# ---- H2: end-of-sequence row of the tied embedding ---------------------------------------------------------------------
model = s.subject.model
eos = model.generation_config.eos_token_id
if eos is None:
    eos = model.config.decoder.eos_token_id
eos = int(eos[0] if isinstance(eos, (list, tuple)) else eos)


def eos_trial(total, base_factor, text, pred):
    """Scale the EOS row so that it ends at ``total`` times its original length; ``base_factor`` is what is already in place."""
    row = s.subject.tensors()["dec.embed.tokens"][[eos]].copy()
    return trial(f"H2 EOS row x{total:.2f}", text, pred, "set_rows: one row of the tied token embedding / readout",
                 "set_rows", ["dec.embed.tokens"], index=[eos], values=(row * (total / base_factor)).astype(np.float32))


s.note(f"EOS id {eos}; row norm {float(np.linalg.norm(s.subject.tensors()['dec.embed.tokens'][eos])):.4f}")
d, kept = eos_trial(0.90, 1.0,
                    "Length ratio 0.958: lines end early; a shorter EOS row in the tied readout lowers the EOS logit (no output bias exists)",
                    "length ratio up about 0.005, CER -0.1 to -0.3 points, risk of more run-on lines; too small for the keep rule")
if d is not None:
    if kept or d < -STEP:
        eos_trial(0.80, 0.90 if kept else 1.0, "EOS row shorter still (the first step lowered the CER)",
                  "a further fall of the same size or less")
    elif d > STEP:
        eos_trial(1.10, 1.0, "EOS row longer (the shorter row raised the CER: the stop decision was not early)",
                  "CER -0.0 to -0.2 points")
    else:
        s.note("H2: change below 0.1 point, no follow-up")

# ---- H3: weight of the image evidence -----------------------------------------------------------------------------------
names_o = [f"dec.{l}.cross.o.{p}" for l in LAYERS for p in ("weight", "bias")]
scale_with_followup("H3 cross.o", names_o,
                    "More image evidence in the decoder stream against a context prior that drops letters: gain of every cross-attention output",
                    "CER -0.1 to +0.3 points, direction uncertain, slightly more likely a rise",
                    "scale: decoder cross-attention output, all layers", 1.15, 1.15, 0.87)

# ---- H4: lengthen the encoder's fine-tune step (extrapolate away from the donor) ------------------------------------------
if s.budget_left >= 1:
    D = TrocrSubject(DONOR, device="cpu", check_card=False).tensors()
    T = s.subject.tensors()
    names_e = [f"enc.{l}.attn.{k}.weight" for l in LAYERS for k in ("q", "k", "v", "o")] + \
              [f"enc.{l}.ffn.{k}.weight" for l in LAYERS for k in ("in", "out")]
    names_e = [n for n in names_e if n in D and n in T and D[n].shape == T[n].shape]
    per = {n: {"other": D[n].copy()} for n in names_e}
    del D
    trial("H4 encoder away from donor a=-0.15",
          "The model is under-fitted on its own training pages; the encoder's step from the donor may be too short. Extrapolate the encoder "
          "attention and FFN matrices 15 % beyond the fine-tune step (interpolate with alpha -0.15; not the toward-donor comparator)",
          "CER +0.0 to +0.5 points (harm more likely than gain)",
          "interpolate (alpha<0) with the donor: encoder attention and FFN weights", "interpolate", names_e, per_tensor=per, alpha=-0.15)
    del per
else:
    s.note("H4: skipped, no pass left")

# ---- H5: logit temperature (with the passes that remain) ---------------------------------------------------------------
names_t = ["dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"]
if s.budget_left >= 1:
    d, kept = trial("H5 logits x1.25",
                    "dec.11.norm_ffn is the last operation before the tied readout: its gain and bias times 1.25 is all logits times 1.25; "
                    "greedy choices are unchanged, beam ranking and the stop decision are not",
                    "|change| below 0.2 point, sign unknown",
                    "scale: final decoder layer norm (temperature)", "scale", names_t, factor=1.25)
    if d is not None and s.budget_left >= 1:
        if kept or d < -STEP:
            trial("H5 logits x1.56 total", "Sharper still (the first step lowered the CER)", "a further fall of the same size or less",
                  "scale: final decoder layer norm (temperature)", "scale", names_t, factor=1.25 if kept else 1.5625)
        elif d > STEP:
            trial("H5 logits x0.80", "Flatter logits (the sharper ones raised the CER)", "CER -0.0 to -0.2 points",
                  "scale: final decoder layer norm (temperature)", "scale", names_t, factor=0.80)
        else:
            s.note("H5: change below 0.1 point, no follow-up")
else:
    s.note("H5: skipped, no pass left")

s.note(f"end: last kept state CER {state['cer']:.4f} (C0 {m0.cer:.4f}), passes left {s.budget_left}")
summary = s.freeze()
print({k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})
