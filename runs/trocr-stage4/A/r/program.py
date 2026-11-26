"""Subject A, task (a), with the donor. Pseudo output biases through the last layer's norm bias, then donor interpolation."""
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

DONOR = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
BIAS = "dec.11.norm_ffn.bias"
EMB = "dec.embed.tokens"

tok = s.subject.tokenizer
V = s.subject.tensors()[EMB].shape[0]


def emb():
    return s.subject.tensors()[EMB].astype(np.float64)


def word_initial_ids():
    ids = []
    for i in range(min(V, len(tok))):
        t = tok.convert_ids_to_tokens(i)
        if isinstance(t, str) and (t.startswith("Ġ") or t.startswith("▁")):
            ids.append(i)
    return np.array(ids, dtype=int)


def bias_delta_words(target):
    """d such that the mean logit shift of word-initial tokens is `target` nats."""
    E = emb()
    W = word_initial_ids()
    if len(W) < 100:
        return None
    v = E[W].mean(axis=0) - E.mean(axis=0)
    shift = (E[W] @ v).mean()
    if shift <= 0:
        return None
    return (v * (target / shift)).astype(np.float32)


def bias_delta_eos(drop):
    E = emb()
    eos = tok.eos_token_id
    if eos is None:
        return None
    e = E[eos]
    d = -e / float(e @ e) * drop            # lowers the EOS logit by `drop` nats (h . d = -drop when h is along e)
    return d.astype(np.float32)


def try_edit(label, text, pred, cls, op, names, base, **kw):
    """One edit, one pass, the rule decides (a rise in collapsed lines also reverts). Returns (kept, metrics)."""
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=MASK)
    if m.collapsed > base.collapsed + 10:
        s.revert(f"collapsed lines rose {base.collapsed} -> {m.collapsed}")
        return False, m
    return s.keep(label), m


base = s.measure("C0", rule_mask=MASK)
s.note(f"C0 cer {base.cer:.4f} collapsed {base.collapsed}")
cur = base

# H1: word-initial prior through the last layer's norm bias
kept_w = 0.0
d = bias_delta_words(0.5)
if d is not None and s.budget_left > 3:
    ok, m = try_edit("H1 word-initial +0.5", "Shift readout toward word-initial tokens (missing spaces)",
                     "CER -0.5 to -1.5 points", "add (pseudo output bias)", "add", [BIAS], cur, delta=d)
    if ok:
        kept_w, cur = 0.5, m
        d2 = bias_delta_words(0.5)
        if d2 is not None:
            ok2, m2 = try_edit("H1b word-initial +0.5 more", "Further word-initial shift",
                               "CER -0.2 to -0.6 further", "add (pseudo output bias)", "add", [BIAS], cur, delta=d2)
            if ok2:
                kept_w, cur = 1.0, m2
    elif m.cer < cur.cer and s.budget_left > 3:
        d3 = bias_delta_words(0.25)
        if d3 is not None:
            ok3, m3 = try_edit("H1c word-initial +0.25", "Smaller word-initial shift (the 0.5 one lowered point CER, not significantly)",
                               "CER -0.2 to -0.5", "add (pseudo output bias)", "add", [BIAS], cur, delta=d3)
            if ok3:
                kept_w, cur = 0.25, m3

# H2: end-of-sequence suppression
d = bias_delta_eos(0.7)
if d is not None and s.budget_left > 3:
    ok, m = try_edit("H2 eos -0.7", "Lower the end-of-sequence logit (short outputs, collapsed lines)",
                     "CER -0.2 to -0.8 points", "add (pseudo output bias)", "add", [BIAS], cur, delta=d)
    if ok:
        cur = m

# H3: partial move toward the donor over all non-embedding tensors
if s.budget_left >= 2:
    D = TrocrSubject(DONOR, device="cpu", check_card=False).tensors()
    names = [n for n in sorted(s.subject.tensors()) if n in D and n != EMB]
    per = {n: {"other": D[n]} for n in names}
    h = s.hypothesis("Interpolate all non-embedding tensors 0.25 toward the donor",
                     "CER change between -0.3 and +0.3 points", "interpolate (global, comparator-like)")
    s.edit("interpolate", names, hypothesis=h, per_tensor=per, alpha=0.25)
    m = s.measure("H3 donor alpha 0.25", rule_mask=MASK)
    if m.collapsed > cur.collapsed + 10:
        s.revert("collapsed rose")
        kept3 = False
    else:
        kept3 = s.keep("H3 donor alpha 0.25")
    if kept3:
        cur = m
        if s.budget_left >= 1:
            per = {n: {"other": D[n]} for n in names}
            h = s.hypothesis("A second step 0.25 toward the donor (total about 0.44)",
                             "CER -0.1 to -0.3 further", "interpolate (global, comparator-like)")
            s.edit("interpolate", names, hypothesis=h, per_tensor=per, alpha=0.25)
            m = s.measure("H3b donor alpha 0.25 again", rule_mask=MASK)
            if m.collapsed > cur.collapsed + 10:
                s.revert("collapsed rose")
            else:
                s.keep("H3b donor alpha 0.25 again")

s.note("end of program")
summary = s.freeze()
print(summary.get("kept"), "edits kept of", summary.get("edits"))
