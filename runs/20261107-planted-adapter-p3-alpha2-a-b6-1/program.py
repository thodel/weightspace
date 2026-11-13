from wsp.run import open_session
import numpy as np

s = open_session(".")
sub = s.subject
Bnames = [k for k in sub.tensors() if k.endswith("lora_B.weight")]
net = 1.0
used = 0


def can():
    return s.budget_left > 0


def try_scale(f, text, pred, label):
    """Scale all B by f, measure, keep by the harness rule. Returns True if kept."""
    global net
    h = s.hypothesis(text, pred, "scale")
    s.edit("scale", Bnames, hypothesis=h, factor=f)
    m = s.measure(label)
    k = s.keep(label)
    ok = bool(getattr(k, "kept", k))
    if ok:
        net *= f
    s.note("%s: f=%s cer=%.3f collapsed=%s kept=%s" % (label, f, m.cer, getattr(m, "collapsed", None), ok))
    return ok


base = s.measure("C0")
s.note("C0 cer=%.3f" % base.cer)

if can():
    ok1 = try_scale(0.5,
                    "lora_alpha doubled (256 vs 128): all B x0.5 restores alpha 128",
                    "CER 2.5 -> 0.3-0.6, at_cap 29 -> <10", "alpha_half")
else:
    ok1 = False

if ok1:
    if can():
        ok2 = try_scale(1.1, "net 0.55 matches norm ratio 1/1.8",
                        "within noise, +-0.05", "net0.55")
    else:
        ok2 = False
    if not ok2 and can():
        try_scale(0.85, "net 0.425 slightly below half",
                  "within noise, +-0.05", "net0.425")
else:
    if can():
        ok3 = try_scale(0.7, "partial over-scale: x0.7 from original",
                        "CER falls by >=1 if scale is partly the defect", "x0.7")
        if ok3 and can():
            try_scale(0.7, "further x0.7 (net 0.49)", "CER falls further", "x0.49")

s.note("net scale on B kept: %.3f" % net)
summary = s.freeze()
