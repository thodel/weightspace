import numpy as np
from wsp.run import open_session

s = open_session(".")
T = s.subject.tensors()
MASK = s.unseen_by()

m0 = s.measure("C0", rule_mask=MASK)
base_collapsed = m0.collapsed


def try_edit(text, pred, cls, op, names, label, reason, **kw):
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=MASK)
    if m.collapsed > base_collapsed + 3:
        s.revert(label + ": collapsed lines up")
        return False
    return bool(s.keep(reason))


# H1: enc.06 attention readout is zero; add damped pinv(v) readout
Wv = T["enc.06.attn.v.weight"].astype(np.float64)
inv_v = np.linalg.pinv(Wv, rcond=1e-3).astype(np.float32)
kept1 = try_edit(
    "enc.06 attn.o is all zero (dead attention); approximate readout 0.3*pinv(v)",
    "CER -0.00..-0.03, p(kept)~0.2; unrecoverable without parent if it fails",
    "add", "add", ["enc.06.attn.o.weight"], "H1 pinv(v)*0.3", "enc.06 readout 0.3 pinv(v)",
    delta=0.3 * inv_v)
if kept1 and s.budget_left > 0:
    try_edit("H1b: second 0.3 of pinv(v) readout", "further -0.00..-0.02", "add", "add",
             ["enc.06.attn.o.weight"], "H1b", "enc.06 readout total 0.6",
             delta=0.3 * inv_v)

# H2: dec.11.norm_ffn inflated
try_edit("dec.11.norm_ffn weight/bias inflated ~2.2x; scale 0.7",
         "CER -0.01..-0.05 if planted, +0.01..+0.05 if natural", "scale", "scale",
         ["dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"], "H2 dec11 norm_ffn x0.7",
         "dec.11.norm_ffn x0.7", factor=0.7)

# H3: enc.10.norm_ffn inflated
try_edit("enc.10.norm_ffn weight/bias inflated ~2.5x; scale 0.7",
         "CER -0.01..-0.05 if planted, worse if natural; p(kept)~0.1", "scale", "scale",
         ["enc.10.norm_ffn.weight", "enc.10.norm_ffn.bias"], "H3 enc10 norm_ffn x0.7",
         "enc.10.norm_ffn x0.7", factor=0.7)

# H4: EOS row (id 2) of the tied embedding slightly too strong
if s.budget_left > 0:
    row = T["dec.embed.tokens"][2].copy()
    try_edit("EOS row scaled 0.95: fewer early stops (length ratio 0.905)",
             "CER -0.005..-0.01 if early EOS matters, else ~0", "set_rows", "set_rows",
             ["dec.embed.tokens"], "H4 EOS x0.95", "EOS row x0.95",
             index=[2], values=(0.95 * row)[None, :])

s.note("done; deletions-dominated C0, enc.06 attn.o zero not restorable without donor")
s.freeze()
