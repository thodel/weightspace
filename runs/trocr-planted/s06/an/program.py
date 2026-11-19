from wsp.run import open_session
import numpy as np

s = open_session(".")
MASK = s.unseen_by()
base = s.measure("C0", rule_mask=MASK)
last = base


def try_edit(name, text, pred, cls, tensors, factor):
    global last
    if s.budget_left <= 0:
        return
    h = s.hypothesis(text, pred, cls)
    s.edit("scale", tensors, hypothesis=h, factor=factor)
    m = s.measure(name, rule_mask=MASK)
    if m.cer < last.cer and m.collapsed <= last.collapsed + 3:
        if s.keep(name):
            last = m
        return
    s.revert(name + ": no gain")


T = s.subject.tensors()

try_edit("H1 enc00 v x1.5", "enc.00 attn.v shrunk", "CER -0.5..-3 points", "scale",
         ["enc.00.attn.v.weight"], 1.5)
try_edit("H2 enc11 norm_attn x1.6", "enc.11 norm_attn.weight shrunk", "CER -0.5..-3 points", "scale",
         ["enc.11.norm_attn.weight"], 1.6)
try_edit("H3 dec11 norm_ffn x0.7", "dec.11 norm_ffn.weight inflated", "CER -0.5..-2 points", "scale",
         ["dec.11.norm_ffn.weight"], 0.7)
try_edit("H4 cross.o x1.15", "cross-attention output too weak", "CER -0.5..-2 points", "scale",
         ["dec.%02d.cross.o.weight" % i for i in range(12)], 1.15)
try_edit("H5 enc ffn.out x1.05", "mild global encoder gain", "CER within +-0.5", "scale",
         ["enc.%02d.ffn.out.%s" % (i, k) for i in range(12) for k in ("weight", "bias")], 1.05)
try_edit("H6 enc final norm x1.1", "final encoder norm scale", "CER within +-0.5", "scale",
         ["enc.norm_final.weight"], 1.1)

s.note("final CER %.4f vs C0 %.4f" % (last.cer, base.cer))
s.freeze()
