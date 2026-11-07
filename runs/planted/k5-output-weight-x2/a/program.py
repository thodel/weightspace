import numpy as np
from wsp.run import open_session

s = open_session(".")
W, B = "O_18.lin.weight", "O_18.lin.bias"


def try_scale(factor, text, pred):
    h = s.hypothesis(text, pred, "scale")
    s.edit("scale", [W], hypothesis=h, factor=factor)
    m = s.measure(f"{W} x{factor}")
    ok = s.keep(f"output weight x{factor}")
    return ok, m


base = s.measure("C0")
best = base.cer
s.note(f"baseline CER {base.cer:.4f} collapsed {base.collapsed}")

ok, m = try_scale(0.5, "output weight was doubled; halve it",
                  "CER 0.065 -> ~0.04 (delta -0.02), fewer collapsed lines")
if ok:
    best = m.cer
    for f in (0.85, 0.85):          # refinement, only while it keeps helping
        if s.budget_left < 1:
            break
        ok2, m2 = try_scale(f, f"refine output scale by x{f}", "small gain -0.002..-0.005")
        if not ok2:
            break
        best = m2.cer
else:
    ok, m = try_scale(0.7, "output weight inflated by less than 2x", "CER -0.005..-0.015")
    if ok:
        best = m.cer
    else:
        ok, m = try_scale(1.3, "wrong direction check", "CER rises; keep only if direction misread")
        if ok:
            best = m.cer

# blank-bias balance, chosen from the latest confusions (deletions dominate -> lower blank bias)
if s.budget_left >= 1:
    dels = 0
    try:
        conf = m.confusions if hasattr(m, "confusions") else []
        for row in conf:
            ref, hyp, cnt = (row if not isinstance(row, dict) else (row.get("ref"), row.get("hyp"), row.get("count")))
            if hyp in ("∅", "", None):
                dels += cnt
            elif ref in ("∅", "", None):
                dels -= cnt
    except Exception:
        dels = 1
    d = np.zeros_like(s.subject.tensors()[B])
    d[0] = -0.5 if dels >= 0 else +0.5
    h = s.hypothesis("blank bias balance", "CER -0.003..-0.01", "add")
    s.edit("add", [B], hypothesis=h, delta=d.astype(np.float32))
    m3 = s.measure("blank bias")
    s.keep("blank bias shift")

summary = s.freeze()
