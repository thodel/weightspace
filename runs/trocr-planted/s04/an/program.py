from wsp.run import open_session
import numpy as np

s = open_session(".")
MASK = s.unseen_by()
base = s.measure("C0", rule_mask=MASK)
best = base.cer
s.note(f"C0 cer={base.cer:.4f} collapsed={base.collapsed}")


def try_edit(text, pred, cls, op, names, **kw):
    global best
    if s.budget_left < 1:
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(text[:40], rule_mask=MASK)
    ok = m.cer < best and m.collapsed <= base.collapsed + 3
    if ok and s.keep(text):
        best = m.cer
        return True
    if not ok:
        s.revert(text)
    return False


dn = ["dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"]
k1 = try_edit("H1 dec.11 norm_ffn x0.5", "CER -2 to -6 points if planted", "scale", "scale", dn, factor=0.5)
if not k1:
    s.note("H1 not kept")
    try_edit("H1b dec.11 norm_ffn x0.75", "CER -1 to -2 points", "scale", "scale", dn, factor=0.75)

en = ["enc.10.norm_ffn.weight", "enc.10.norm_ffn.bias"]
k2 = try_edit("H2 enc.10 norm_ffn x0.6", "CER -1 to -3 points if planted", "scale", "scale", en, factor=0.6)

try_edit("H3 enc.00 attn.v x1.4", "CER -0.5 to -1.5 points if shrunk", "scale", "scale",
         ["enc.00.attn.v.weight"], factor=1.4)

if k2:
    try_edit("H4 enc.10 ffn.out x0.85", "CER -0.5 point", "scale", "scale",
             ["enc.10.ffn.out.weight", "enc.10.ffn.out.bias"], factor=0.85)

s.freeze()
