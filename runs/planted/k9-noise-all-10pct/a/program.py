import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
s = open_session(".")
base = s.measure("C0")
best = base.cer
state = {"cer": best}


def attempt(text, pred, cls, op, names, label, **kw):
    if s.budget_left < 1:
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label)
    ok = s.keep(label)
    if ok:
        state["cer"] = m.cer
    s.note(f"{label}: cer {m.cer:.4f} collapsed {m.collapsed} kept {ok}")
    return ok


def bias_delta(n_out, row0, d, others=0.0):
    v = np.full(n_out, others, dtype=np.float32)
    v[0] = d
    return v


T = s.subject.tensors()
n = T["O_18.lin.bias"].shape[0]

# H1/H2: lower blank bias
ok1 = attempt("blank bias too high, chars deleted", "CER -0.10..-0.2", "output bias",
              "add", ["O_18.lin.bias"], "H1 blank bias -1.5", delta=bias_delta(n, 0, -1.5))
step = -1.5 if ok1 else -0.75
ok2 = attempt("blank still dominant", "CER -0.03..-0.08", "output bias",
              "add", ["O_18.lin.bias"], "H2 blank bias further", delta=bias_delta(n, 0, step))

# H3: raise all char biases (only if blank bias moves helped, else try anyway)
attempt("char biases too negative", "CER -0.02", "output bias", "add", ["O_18.lin.bias"],
        "H3 char bias +0.5", delta=bias_delta(n, 0, 0.0, 0.5) + np.r_[0.0, np.zeros(n - 1)].astype(np.float32))

# H4: C_3 bias to parent
P = KrakenSubject(PARENT).tensors()
attempt("C_3 bias is damaged", "CER -0.02..-0.05", "donor interpolate", "interpolate",
        ["C_3.co.bias"], "H4 C_3 bias to parent", other=P["C_3.co.bias"], alpha=1.0)

# H5: convs 25% toward parent
convs = [f"C_{i}.co.{k}" for i in (0, 3, 6, 9) for k in ("weight", "bias")]
attempt("noise averaged by parent", "CER -0.02..-0.05", "donor interpolate", "interpolate", convs,
        "H5 convs 25% parent", per_tensor={c: {"other": P[c]} for c in convs}, alpha=0.25)

# H6: LSTMs 25% toward parent
ls = [k for k in T if k.startswith("L_") and "bias_ih" not in k]
attempt("noise averaged by parent", "CER -0.02", "donor interpolate", "interpolate", ls,
        "H6 lstm 25% parent", per_tensor={c: {"other": P[c]} for c in ls}, alpha=0.25)

# H7: blank-row scale, direction from remaining deletion share
if s.budget_left >= 1:
    m = s.measure("probe") if False else None
    attempt("blank row too strong", "CER -0.02", "row scale", "set_rows", ["O_18.lin.weight"],
            "H7 blank row x0.85", index=[0], values=(s.subject.tensors()["O_18.lin.weight"][0:1] * 0.85))

summary = s.freeze()
