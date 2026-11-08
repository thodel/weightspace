from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
s = open_session(".")
P = KrakenSubject(PARENT).tensors()
L14W = [f"L_14.layer.{k}{r}" for r in ("", "_reverse") for k in ("weight_ih_l0", "weight_hh_l0")]
L14B = [f"L_14.layer.bias_hh_l0{r}" for r in ("", "_reverse")]


def donors(names):
    return {n: {"other": P[n].copy()} for n in names}


def try_edit(text, pred, cls, op, names, label, **kw):
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label)
    ok = s.keep(label)
    return ok, m


base = s.measure("C0")
best = base.cer
s.note(f"baseline CER {base.cer:.4f}, collapsed {base.collapsed}")

ok, m = try_edit("L_14 weights carry noise; pull 40% toward parent", "CER -0.02 to -0.03",
                 "interpolate", "interpolate", L14W, "L14 alpha .4",
                 per_tensor=donors(L14W), alpha=0.4)
if ok:
    best = m.cer
    # remainder step: total shift ~0.7 of the way
    ok2, m2 = try_edit("further pull, 0.5 of remainder", "CER -0.005 to -0.015",
                       "interpolate", "interpolate", L14W, "L14 +alpha .5",
                       per_tensor=donors(L14W), alpha=0.5)
    if ok2:
        best = m2.cer
else:
    ok1b, m1b = try_edit("smaller pull alpha 0.2", "CER -0.01", "interpolate", "interpolate",
                         L14W, "L14 alpha .2", per_tensor=donors(L14W), alpha=0.2)
    if ok1b:
        best = m1b.cer
    else:
        ok3, m3 = try_edit("noise inflates norm: scale L_14 weights 0.9", "CER -0.005", "scale",
                           "scale", L14W, "L14 x0.9", factor=0.9)
        if ok3:
            best = m3.cer

C3b = ["C_3.co.bias"]
try_edit("C_3 bias far from parent; pull halfway", "CER 0 to -0.005", "interpolate",
         "interpolate", C3b, "C3 bias .5", per_tensor=donors(C3b), alpha=0.5)

if s.budget_left > 0:
    try_edit("L_14 hh biases pull halfway", "CER change < 0.003", "interpolate",
             "interpolate", L14B, "L14 bias .5", per_tensor=donors(L14B), alpha=0.5)

print(s.freeze())
