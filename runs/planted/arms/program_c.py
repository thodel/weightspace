"""Arm (c): a hand-written list of the obvious repairs, the same for every corrupted model (written 07.11.2026 before any pass).

Each rule looks only at the model's tensors against its parent kraken-german-xix-v1 and proposes one edit; the harness keep
rule (paired interval below 0) decides. Thresholds are set from the family's healthy fine-tunes (docs/readings): output rows of
shared characters keep cosine >= 0.86 to the parent, blank-minus-mean-bias gap moves < 0.2, block norms stay within 5 % of the parent's,
relative block distances are 0.12-0.29.
"""
import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
s = open_session(".")
P = KrakenSubject(PARENT); pt = P.tensors(); pc = P.codec_map()
W, B = "O_18.lin.weight", "O_18.lin.bias"
s.measure("C0")


def T(): return s.subject.tensors()


def try_edit(text, pred, cls, op, names, **kw):
    if s.budget_left < 1:
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    s.measure(text[:60])
    return s.keep(text[:60])


# R1: output rows whose direction disagrees with the parent's (zeroed, negated, swapped): copy parent row and bias, by character
cm = s.subject.codec_map(); w, b = T()[W], T()[B]
bad = []
for c, r in cm.items():
    if c in pc:
        a, p = w[r], pt[W][pc[c]]
        cos = float(a @ p / ((np.linalg.norm(a) * np.linalg.norm(p)) or 1e-9))
        if cos < 0.5:
            bad.append(c)
if bad:
    rows = [cm[c] for c in bad]
    h = s.hypothesis(f"R1 rows with cosine < 0.5 to the parent ({bad}) are broken: copy parent weights and biases",
                     "CER falls toward the healthy level", "output rows from parent")
    s.edit("set_rows", [W, B], hypothesis=h, index=rows,
           per_tensor={W: {"values": pt[W][[pc[c] for c in bad]].astype(w.dtype)}, B: {"values": pt[B][[pc[c] for c in bad]].astype(b.dtype)}})
    s.measure("R1")
    s.keep("R1")

# R2: the blank's bias relative to the mean of the others against the parent's
b = T()[B]; gap = float(b[0] - b[1:].mean()); pgap = float(pt[B][0] - pt[B][1:].mean())
if abs(gap - pgap) > 1.0 and s.budget_left >= 1:
    d = np.zeros_like(b); d[0] = pgap - gap
    try_edit(f"R2 blank-minus-mean bias gap {gap:.2f} vs parent {pgap:.2f}: restore the parent's gap", "CER falls", "output bias (blank)", "add", [B], delta=d)

# R3: blocks whose weight norm left the parent's by more than 5 %: rescale to the parent's norm (skips zeroed blocks)
blocks = {}
for n, v in T().items():
    if n.startswith("O_") or n not in pt or pt[n].shape != v.shape:
        continue
    blocks.setdefault(n.split(".")[0], []).append(n)
for blk, names in blocks.items():
    if s.budget_left < 3: break
    own = np.sqrt(sum(float((T()[n].astype(np.float64) ** 2).sum()) for n in names))
    par = np.sqrt(sum(float((pt[n].astype(np.float64) ** 2).sum()) for n in names))
    if own > 0 and not (0.95 <= own / par <= 1.05):
        try_edit(f"R3 block {blk} has norm {own/par:.2f} x the parent's: rescale", "CER falls", f"block {blk} scale", "scale", names, factor=float(par / own))

# R4: the block farthest from the parent (relative L2) when it is > 1.3 x the median block distance: transplant, else halfway
def rel(blk):
    names = blocks[blk]
    d = np.sqrt(sum(float(((T()[n] - pt[n]).astype(np.float64) ** 2).sum()) for n in names))
    return d / np.sqrt(sum(float((pt[n].astype(np.float64) ** 2).sum()) for n in names))
for _ in range(2):
    if s.budget_left < 1: break
    r = {b_: rel(b_) for b_ in blocks}; med = float(np.median(list(r.values()))); top = max(r, key=r.get)
    if r[top] <= 1.3 * med: break
    names = blocks[top]
    ok = try_edit(f"R4 block {top} is {r[top]/med:.2f} x the median distance from the parent: transplant it", "CER falls", f"block {top} from parent",
                  "transplant", names, per_tensor={n: {"donor": pt[n]} for n in names})
    if not ok and s.budget_left >= 1:
        try_edit(f"R4b block {top}: halfway to the parent", "CER falls", f"block {top} interpolate", "interpolate", names,
                 per_tensor={n: {"other": pt[n]} for n in names}, alpha=0.5)
        break
s.note("hand-written repair list (arm c) done; unspent passes left unspent")
s.freeze()
