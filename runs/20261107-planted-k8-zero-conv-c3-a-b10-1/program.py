import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
s = open_session(".")
P = KrakenSubject(PARENT).tensors()
P = {k: np.array(v, copy=True) for k, v in P.items()}

m = s.measure("C0")
cur = m.cer
s.note(f"baseline CER {cur:.4f} collapsed {m.collapsed}")

def attempt(text, pred, cls, op, names, why, **kw):
    """returns True if kept. Needs a pass; skips if none left."""
    global cur, m
    if s.budget_left < 1:
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m2 = s.measure(why)
    if s.keep(why):
        cur, m = m2.cer, m2
        return True
    return False

# H1: transplant C_3 from parent
c3 = ["C_3.co.weight", "C_3.co.bias"]
ok1 = attempt("C_3 is zeroed; restore parent's C_3", "CER 1.00 -> 0.3-0.6", "transplant",
              "transplant", c3, "C_3 from parent",
              per_tensor={n: {"donor": P[n]} for n in c3})
if not ok1:
    s.note("H1 not kept; stop, nothing else tried without a working base")
else:
    # H2: pull C_6 toward parent
    c6 = ["C_6.co.weight", "C_6.co.bias"]
    ok2 = attempt("child C_6 co-adapted to lost C_3; interpolate to parent 0.5", "-0.05..-0.2", "interpolate",
                  "interpolate", c6, "C_6 alpha .5",
                  per_tensor={n: {"other": P[n]} for n in c6}, alpha=0.5)
    c9 = ["C_9.co.weight", "C_9.co.bias"]
    ok3 = attempt("same for C_9, alpha 0.5", "-0.02..-0.1", "interpolate",
                  "interpolate", c9, "C_9 alpha .5",
                  per_tensor={n: {"other": P[n]} for n in c9}, alpha=0.5)
    if ok2 or ok3:
        # H3: go full transplant on whichever helped
        for names, flag, tag in ((c6, ok2, "C_6"), (c9, ok3, "C_9")):
            if flag:
                attempt(f"full transplant {tag}", "-0.02..-0.1", "transplant",
                        "transplant", names, f"{tag} alpha 1",
                        per_tensor={n: {"donor": P[n]} for n in names})
    # H4: scale C_3 weight
    for f in (1.15, 0.85):
        if s.budget_left >= 3:
            if attempt(f"C_3 weight scale {f}", "within +-0.02", "scale",
                       "scale", ["C_3.co.weight"], f"C_3 scale {f}", factor=f):
                break
    # H5: blank bias
    if s.budget_left >= 1 and m.collapsed > 0:
        d = np.zeros(297, dtype=np.float32)
        d[0] = -0.5
        attempt("blank bias too high", "-0.01..-0.05", "add",
                "add", ["O_18.lin.bias"], "blank bias -0.5", delta=d)

summary = s.freeze()
