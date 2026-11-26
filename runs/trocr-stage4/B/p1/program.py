from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

DONOR = "/home/tobias/atr-cache/trocr/microsoft__trocr-base-handwritten"

s = open_session(".")
T = s.subject.tensors()
D = TrocrSubject(DONOR, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-base-handwritten")

base = s.measure("C0", rule_mask=MASK)
cur = base.cer
s.note(f"C0 CER {base.cer:.4f} collapsed {base.collapsed}")


def attempt(label, text, pred, cls, op, names, **kw):
    """One edit, one pass, keep by the harness rule. Returns (kept, cer)."""
    global cur
    if s.budget_left < 1:
        return False, None
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=MASK)
    s.note(f"{label}: CER {m.cer:.4f} (state before {cur:.4f}) collapsed {m.collapsed}")
    kept = s.keep(label)
    if kept:
        cur = m.cer
    return kept, m.cer


COB = [f"dec.{i:02d}.cross.o.bias" for i in range(12)]

# H1: cross.o.bias magnitude
k, c = attempt("H1a cross.o.bias x0.9",
               "dec cross.o.bias norms are 2.2-2.9x donor: possible overshoot",
               "CER -0.001..+0.004, probably slightly worse", "scale", "scale", COB, factor=0.9)
if not k and c is not None and c > cur + 0.002:
    attempt("H1b cross.o.bias x1.1",
            "the other side of the cross.o.bias magnitude",
            "CER -0.002..+0.004", "scale", "scale", COB, factor=1.1)

# H2: EOS row (id 2 of the tied embedding) sharpened
attempt("H2 eos row x1.05",
        "deletions/short outputs: sharpen the EOS logit both ways",
        "CER -0.002..+0.002", "scale", "set_rows", ["dec.embed.tokens"],
        index=[2], values=(T["dec.embed.tokens"][2] * 1.05).astype(np.float32)[None, :])

# H3: enc.norm_final.bias magnitude
attempt("H3 enc.norm_final.bias x0.9",
        "most moved encoder tensor, 2.6x donor norm",
        "CER -0.001..+0.005", "scale", "scale", ["enc.norm_final.bias"], factor=0.9)

# H4: tied embedding a small step toward the donor
attempt("H4 dec.embed.tokens toward donor 0.1",
        "embedding holds 71 % of the move; test overshoot",
        "CER +0.002..+0.02 (worse)", "interpolate", "interpolate", ["dec.embed.tokens"],
        alpha=0.1, per_tensor={"dec.embed.tokens": {"other": D["dec.embed.tokens"]}})

s.freeze()
