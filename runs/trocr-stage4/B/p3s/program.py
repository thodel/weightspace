from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject  # noqa: F401  (no donor is used)
import numpy as np

s = open_session(".")
MASK = s.unseen_by()

base = s.measure("C0", rule_mask=MASK)
base_collapsed = base.collapsed
state = {"collapsed": base_collapsed}


def attempt(label, text, pred, cls, op, names, **kw):
    """One edit + one pass + keep/revert. Returns True when kept."""
    if s.budget_left < 1:
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=MASK)
    if m.collapsed > state["collapsed"] + 10:
        s.revert(f"{label}: collapsed lines rose {state['collapsed']} -> {m.collapsed}")
        return False
    if m.cer > base.cer + 0.02:
        s.revert(f"{label}: CER clearly worse")
        return False
    ok = s.keep(label)
    if ok:
        state["collapsed"] = m.collapsed
    return ok


def scale_pair(label, text, pred, names, f):
    """Try factor f; if kept, repeat once; if refused, try the inverse factor once."""
    if attempt(f"{label} x{f}", text, pred, "scale", "scale", names, factor=f):
        attempt(f"{label} x{f} again", text + " (repeat)", pred, "scale", "scale", names, factor=f)
    else:
        g = round(1.0 / f, 4)
        attempt(f"{label} x{g}", text + " (opposite direction)", pred, "scale", "scale", names, factor=g)


# H1 output temperature
scale_pair("H1 final LN gain", "Final decoder LayerNorm gain scales all logits: sharper output may cut early drops",
           "CER -0.3..+0.3 points", ["dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"], 1.1)

# H2 sharper cross-attention in upper decoder
cq = []
for i in range(6, 12):
    cq += [f"dec.{i:02d}.cross.q.weight", f"dec.{i:02d}.cross.q.bias"]
scale_pair("H2 cross.q 6-11", "Sharper cross-attention in dec 6-11 locks onto the next glyph",
           "CER -0.2..+0.5 points", cq, 1.1)

# H3 encoder output gain
scale_pair("H3 enc.norm_final", "Encoder output gain: stronger image evidence and sharper cross keys",
           "CER -0.2..+0.3 points", ["enc.norm_final.weight", "enc.norm_final.bias"], 1.05)

# H4 more image into the decoder residual
co = []
for i in range(7, 12):
    co += [f"dec.{i:02d}.cross.o.weight", f"dec.{i:02d}.cross.o.bias"]
if s.budget_left >= 1:
    attempt("H4 cross.o 7-11 x1.1", "More weight on image evidence against the language prior, to counter deletions",
            "CER -0.3..+0.5 points", "scale", "scale", co, factor=1.1)

# H5 EOS row, tied embedding row 2
if s.budget_left >= 1:
    T = s.subject.tensors()
    row = (T["dec.embed.tokens"][2] * 0.9).astype(np.float32)[None, :]
    attempt("H5 EOS row x0.9", "Weaker EOS readout (and start-token input) against early stops",
            "CER -0.3..+0.3 points", "set_rows", ["dec.embed.tokens"], index=[2], values=row)

s.note("end of program; every refused or reverted edit was undone by the harness")
summary = s.freeze()
