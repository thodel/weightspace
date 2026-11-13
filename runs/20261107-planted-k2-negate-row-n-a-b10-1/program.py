import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
s = open_session(".")
cm = s.subject.codec_map()
n = cm["n"]
m0 = s.measure("C0")
best = m0.cer
s.note(f"C0 cer {m0.cer:.4f} collapsed {m0.collapsed}")

# H1: negate the row of n
T = s.subject.tensors()
h = s.hypothesis("output row n is sign-flipped (cos -0.99 to parent, norm same)",
                 "CER 0.75 -> ~0.3, collapsed lines < 10", "set_rows")
s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h, index=[n], values=-T["O_18.lin.weight"][[n]])
m1 = s.measure("H1 negate n")
if s.keep("negated output row n"):
    best = m1.cer
else:
    # fallback: transplant the parent's row (weights and bias) in case negation was not exact
    h = s.hypothesis("negation alone insufficient: take parent's n row", "CER down > 0.1", "set_rows donor")
    P = KrakenSubject(PARENT)
    pn = P.codec_map()["n"]
    PT = P.tensors()
    s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h, index=[n], values=PT["O_18.lin.weight"][[pn]])
    m1 = s.measure("H1b parent row n")
    if s.keep("parent row n"):
        best = m1.cer

# H2: bias of n from parent
P = KrakenSubject(PARENT)
pn = P.codec_map()["n"]
pb = P.tensors()["O_18.lin.bias"][pn]
cur = s.subject.tensors()["O_18.lin.bias"][n]
h = s.hypothesis("n bias drifted by 0.14 from parent", "CER change about -0.005, likely refused", "set_rows bias")
s.edit("set_rows", ["O_18.lin.bias"], hypothesis=h, index=[n], values=np.array([pb], dtype=np.float32))
m2 = s.measure("H2 bias n")
if s.keep("n bias from parent"):
    best = m2.cer

# H3: only if still bad
if best > 0.30:
    T = s.subject.tensors()
    donor = KrakenSubject(PARENT).tensors()["C_3.co.bias"]
    h = s.hypothesis("C_3 bias is a second defect (rel 1.09, cos 0.68)", "CER change 0 to +0.02", "interpolate")
    s.edit("interpolate", ["C_3.co.bias"], hypothesis=h, other=donor, alpha=0.5)
    m3 = s.measure("H3 C3 bias half to parent")
    if s.keep("C_3 bias halfway to parent"):
        best = m3.cer

# H4: remaining n confusions -> interpolate the n row toward the parent's
if best > 0.15:
    Pt = KrakenSubject(PARENT).tensors()
    T = s.subject.tensors()
    row = T["O_18.lin.weight"][n]
    tgt = Pt["O_18.lin.weight"][pn]
    h = s.hypothesis("row n remains off after fix", "CER -0.01 to -0.03", "interpolate row")
    s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h, index=[n], values=(0.5 * row + 0.5 * tgt)[None, :].astype(np.float32))
    m4 = s.measure("H4 row n halfway to parent")
    s.keep("n row halfway to parent")

print(s.freeze())
