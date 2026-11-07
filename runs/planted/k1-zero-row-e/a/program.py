"""Programmer (a), budget 10: restore the zeroed output row of `e`."""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"

s = open_session(".")
cm = s.subject.codec_map()
par = KrakenSubject(PARENT)
pm = par.codec_map()
PT = par.tensors()
pw, pb = np.array(PT[W]), np.array(PT[B])
E = cm["e"]
pe = pm["e"]

m = s.measure("C0")
s.note(f"C0 cer={m.cer:.4f} collapsed={m.collapsed}")


def cur():
    T = s.subject.tensors()
    return np.array(T[W][E]), float(T[B][E])


def try_row(text, pred, w_new, b_new, label):
    h = s.hypothesis(text, pred, "set_rows")
    s.edit("set_rows", [W, B], hypothesis=h,
           per_tensor={W: {"index": [E], "values": np.asarray(w_new, dtype=np.float32)[None, :]},
                       B: {"index": [E], "values": np.asarray([b_new], dtype=np.float32)}})
    mm = s.measure(label)
    k = s.keep(label)
    s.note(f"{label}: cer={mm.cer:.4f} collapsed={mm.collapsed} kept={k}")
    return k, mm


# H1: parent row, verbatim
k1, m1 = try_row("Restore parent row and bias of `e`", "CER 0.162 -> 0.05-0.08",
                 pw[pe], pb[pe], "H1 parent e row")

# H2: ridge-mapped parent row (mapping fitted on shared characters, correction to identity)
shared = [c for c in cm if c in pm and c != "e" and c != "<blank>"]
X = np.stack([pw[pm[c]] for c in shared]).astype(np.float64)
Y = np.stack([np.array(s.subject.tensors()[W][cm[c]]) for c in shared]).astype(np.float64)
G = X @ X.T
lam = 0.1 * np.trace(G) / len(G)
dM = X.T @ np.linalg.solve(G + lam * np.eye(len(G)), Y - X)
mapped = pw[pe].astype(np.float64) + pw[pe].astype(np.float64) @ dM
if k1:
    # try it against the kept H1 state
    k2, m2 = try_row("Ridge-mapped parent row for `e` into the child's hidden basis",
                     "further CER drop 0.00-0.01 vs H1", mapped, pb[pe], "H2 mapped e row")
else:
    k2, m2 = try_row("H1 refused; try the ridge-mapped parent row for `e`", "CER drop 0.08-0.10",
                     mapped, pb[pe], "H2 mapped e row (after H1 refused)")
    if not k2:
        # fallback: parent row at a larger scale (a weak row may be too quiet)
        w0 = pw[pe] * 1.3
        try_row("Parent row of `e` scaled 1.3", "CER drop if the row was too quiet", w0, pb[pe], "H2b parent x1.3")

if k1 or k2:
    # H3: scale of the e weight row
    w, b = cur()
    kk, _ = try_row("Scale `e` row by 1.15", "+-0.002", w * 1.15, b, "H3a scale 1.15")
    if not kk:
        w, b = cur()
        try_row("Scale `e` row by 0.87", "+-0.002", w * 0.87, b, "H3b scale 0.87")
    # H4: bias of e
    w, b = cur()
    kk, _ = try_row("Bias of `e` +0.5", "+-0.003", w, b + 0.5, "H4a bias +0.5")
    if not kk:
        w, b = cur()
        try_row("Bias of `e` -0.5", "+-0.003", w, b - 0.5, "H4b bias -0.5")

summary = s.freeze()
print(summary)
