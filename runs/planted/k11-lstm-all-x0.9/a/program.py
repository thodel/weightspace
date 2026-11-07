from wsp.run import open_session
import numpy as np

s = open_session(".")
T = s.subject.tensors()
lstm_all = [n for n in T if n.startswith("L_")]
lstm_w = [n for n in lstm_all if "weight" in n]
lstm_nonzero = [n for n in lstm_all if not n.startswith("bias_ih") and "bias_ih" not in n]
INV = 1.0 / 0.9

base = s.measure("C0")
kept_any = False


def try_edit(text, pred, cls, op, names, reason, **kw):
    """Apply, measure, keep by rule. Needs 2 passes budget check."""
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(text[:40])
    ok = s.keep(reason)
    return ok, m


if s.budget_left >= 1:
    ok, m = try_edit("H1 LSTM all tensors were scaled 0.9: undo", "CER down 0.02-0.03", "scale",
                     "scale", lstm_nonzero, "undo uniform 0.9 shrink", factor=INV)
    kept_any = ok
    if not ok and s.budget_left >= 1:
        ok, m = try_edit("H2 undo shrink on LSTM weights only", "CER down 0.01-0.03", "scale",
                         "scale", lstm_w, "weights x1.111", factor=INV)
        kept_any = ok
    if kept_any and s.budget_left >= 1:
        ok, m = try_edit("H3 extra x1.05 on the LSTMs", "within +-0.003", "scale",
                         "scale", lstm_nonzero if True else lstm_w, "further gain", factor=1.05)
        if not ok:
            s.note("optimum near 1/0.9")
    if s.budget_left >= 1:
        cm = s.subject.codec_map()
        d = np.zeros(297, dtype=np.float32)
        d[0] = -0.5
        ok, m = try_edit("H4 lower blank bias by 0.5", "CER down 0.005-0.01", "add",
                         "add", ["O_18.lin.bias"], "blank bias", delta=d)
    if not kept_any and s.budget_left >= 1:
        ok, m = try_edit("H5 scale non-blank output rows x1.05", "CER down 0.005", "scale",
                         "scale", ["O_18.lin.weight"], "output gain", factor=1.05)
    if kept_any and s.budget_left >= 1:
        from wsp.kraken_subject import KrakenSubject
        PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
        P = KrakenSubject(PARENT).tensors()
        h = s.hypothesis("H6 pull LSTM weights 10 % toward parent", "neutral or worse", "interpolate")
        per = {n: {"other": P[n]} for n in lstm_w}
        s.edit("interpolate", lstm_w, hypothesis=h, per_tensor=per, alpha=0.1)
        s.measure("H6")
        s.keep("parent pull")

if s._last_edit is not None:
    s.revert("undecided")
print(s.freeze())
