from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")
L16 = [f"L_16.layer.{k}{r}" for r in ("", "_reverse")
       for k in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")]
L16W = [n for n in L16 if "weight" in n]
used = 0
base = s.measure("C0"); used += 1
best = base.cer


def try_edit(text, pred, cls, op, names, reason, **kw):
    """one edit, one pass, decided by the harness keep rule. returns (kept, cer)."""
    global used, best
    if used >= 10:
        return False, best
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(text[:40]); used += 1
    kept = s.keep(reason)
    if kept:
        best = m.cer
    s.note(f"{text}: cer {m.cer:.4f} collapsed {m.collapsed} kept {kept}")
    return kept, m.cer


# H1: undo a x0.5 shrink on the last BiLSTM
k1, _ = try_edit("H1 L_16 all tensors x2 (undo planted x0.5)",
                 "CER 0.199 -> ~0.10", "scale", "scale", L16,
                 "undo planted shrink", factor=2.0)
if not k1:
    # H1b: weights only
    k1, _ = try_edit("H1b L_16 weights only x2", "delta -0.05", "scale", "scale", L16W,
                     "weights only", factor=2.0)

if k1:
    # H1c: refine the factor, at most two tries, only the first that is kept stays
    for f in (1.15, 0.9):
        k, _ = try_edit(f"H1c L_16 extra x{f}", "delta -0.005", "scale", "scale",
                        L16 if True else L16W, "refine factor", factor=f)
        if k:
            break

# H3: residual blank preference, only if still poor
if best > 0.06 and used < 9:
    for d in (-0.5, -1.0):
        delta = np.zeros_like(s.subject.tensors()["O_18.lin.bias"])
        delta[0] = d
        k, _ = try_edit(f"H3 blank bias {d:+}", "delta -0.01", "add", "add",
                        ["O_18.lin.bias"], "lower blank bias", delta=delta)
        if not k:
            break

# H4: C_3 bias toward parent (expected no effect)
if used < 10 and best > 0.05:
    try:
        pt = KrakenSubject(PARENT).tensors()["C_3.co.bias"]
        try_edit("H4 C_3.co.bias halfway to parent", "delta ~0", "interpolate", "interpolate",
                 ["C_3.co.bias"], "bias drift", other=pt, alpha=0.5)
    except Exception as e:
        s.note(f"H4 skipped: {e}")

print(s.freeze())
