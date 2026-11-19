from wsp.run import open_session

s = open_session(".")
MASK = s.unseen_by()
T = s.subject.tensors()


def try_edit(label, text, pred, cls, op, names, **kw):
    """Hypothesis, edit, measure, keep (rule enforced) or revert. Returns kept."""
    names = [n for n in names if n in T]
    if not names or s.budget_left < 1:
        s.note("skipped " + label)
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=MASK)
    s.note("%s cer=%.4f collapsed=%s" % (label, m.cer, m.collapsed))
    if m.cer < base["cer"] - 0.005 and m.collapsed <= base["collapsed"] + 3:
        if s.keep(label):
            base["cer"], base["collapsed"] = m.cer, m.collapsed
            return True
    else:
        s.note(label + ": own check failed, reverting")
    s.revert(label)
    return False


m0 = s.measure("C0", rule_mask=MASK)
base = {"cer": m0.cer, "collapsed": m0.collapsed}
s.note("C0 cer=%.4f collapsed=%s" % (m0.cer, m0.collapsed))

dec11 = ["dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"]
k1 = try_edit("H1", "dec.11.norm_ffn gain inflated about 2x (planted scale)",
              "CER -1 to -5 points", "scale", "scale", dec11, factor=0.5)
if not k1:
    try_edit("H1b", "only the dec.11.norm_ffn weight is inflated",
             "CER -0.5 to -1 points", "scale", "scale", dec11[:1], factor=0.7)

try_edit("H2", "enc.10.norm_ffn over-amplified",
         "CER -0.5 to -3 points if planted", "scale", "scale",
         ["enc.10.norm_ffn.weight", "enc.10.norm_ffn.bias"], factor=0.6)

cross = []
for i in range(12):
    cross += ["dec.%02d.cross.o.weight" % i, "dec.%02d.cross.o.bias" % i]
try_edit("H3", "cross-attention outputs diffusely attenuated, deletions of chars",
         "CER -0.5 to -1.5 points", "scale", "scale", cross, factor=1.15)

try_edit("H4", "encoder final norm slightly under-scaled (control)",
         "CER within +-0.5 points", "scale", "scale",
         ["enc.norm_final.weight"], factor=1.1)

summary = s.freeze()
