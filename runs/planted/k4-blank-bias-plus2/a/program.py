import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
s = open_session(".")
m0 = s.measure("C0")
best = m0.cer
s.note(f"C0 cer={m0.cer:.4f} collapsed={m0.collapsed}")


def blank_shift(d, label, pred):
    global best
    delta = np.zeros(s.subject.tensors()["O_18.lin.bias"].shape, dtype=np.float32)
    delta[0] = d
    h = s.hypothesis(f"blank bias shift {d:+.2f}: blank too strong, deletions dominate", pred, "add")
    s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=delta)
    m = s.measure(label)
    ok = s.keep(label)
    if ok:
        best = m.cer
    s.note(f"{label}: cer={m.cer:.4f} collapsed={m.collapsed} kept={ok}")
    return ok


ok = blank_shift(-2.1, "blank-2.1", "CER down by 0.03 or more")
if ok:
    blank_shift(-0.5, "blank-0.5more", "CER down 0.002 to 0.005")
    if s.budget_left > 0:
        # a failed overshoot is undone by the harness; try a milder extra step only if budget is left
        pass
else:
    if blank_shift(-1.0, "blank-1.0", "CER down 0.01 to 0.03"):
        blank_shift(-1.0, "blank-1.0more", "CER down a further 0.005 to 0.02")
    else:
        blank_shift(-3.0, "blank-3.0", "CER down 0.01 to 0.03 if the sign is right")

# H2: C_3 bias toward parent, half way
if s.budget_left > 0:
    try:
        pt = KrakenSubject(PARENT).tensors()["C_3.co.bias"]
        h = s.hypothesis("C_3.co.bias drifted far from parent", "abs change in CER below 0.005", "interpolate")
        s.edit("interpolate", ["C_3.co.bias"], hypothesis=h, other=np.asarray(pt, dtype=np.float32), alpha=0.5)
        m = s.measure("C3bias-half-parent")
        ok = s.keep("C3 bias half to parent")
        s.note(f"C3 bias: cer={m.cer:.4f} kept={ok}")
    except Exception as e:
        s.note(f"H2 skipped: {e!r}")

print(s.freeze())
