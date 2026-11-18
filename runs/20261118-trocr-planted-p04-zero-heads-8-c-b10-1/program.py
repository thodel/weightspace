"""Arm (c): a hand-written list of the obvious repairs, the same for every planted model (written 18.11.2026 before any pass).

Each rule looks only at the model's tensors against its parent donor `trocr-kurrent-XVI-XVII` and proposes one edit; the harness keep rule
(page-clustered paired interval below -min_effect) decides. Thresholds are set from the healthy step `xvi-xvii` -> `from-kurrent-v2`
(docs/trocr/readings.json, per tensor): norm ratio to the parent 0.98-1.07 (weights 1.00-1.03), relative distance of a weight matrix
0.07-0.24 (decoder FFN highest), of the tied embedding 0.53, of a layer norm vector 0.007-0.13; embedding rows keep a cosine well above 0.5.
The rules need the parent donor; the donor-free baseline is arm (d).
"""
import numpy as np
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
s = open_session(".")
pt = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
s.measure("C0", rule_mask=MASK)


def T():
    return s.subject.tensors()


def try_edit(text, pred, cls, op, names, **kw):
    if s.budget_left < 1:
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    s.measure(text[:60], rule_mask=MASK)
    return s.keep(text[:60])


def blocks():
    """name -> (side, layer, part): the tensors of one part of one layer are one block (weight and bias together)."""
    out = {}
    for n in T():
        p = n.split(".")
        if p[0] in ("enc", "dec") and p[1].isdigit():
            out.setdefault((p[0], p[1], p[2]), []).append(n)
    return out


def norm(x):
    return float(np.sqrt((x.astype(np.float64) ** 2).sum()))


# R1: tied-embedding rows with a direction far from the parent's (zeroed, negated, swapped): copy the parent's rows
if True:
    w, pw = T()["dec.embed.tokens"], pt["dec.embed.tokens"]
    nw, npw = np.linalg.norm(w, axis=1), np.linalg.norm(pw, axis=1)
    cos = (w * pw).sum(1) / np.where(nw * npw > 0, nw * npw, 1.0)
    bad = np.where((cos < 0.5) | (nw == 0))[0]
    if len(bad):
        try_edit(f"R1 {len(bad)} embedding rows have cosine < 0.5 (or norm 0) to the parent: copy the parent's rows",
                 "CER falls toward the healthy level", "token rows from parent", "set_rows", ["dec.embed.tokens"],
                 index=[int(i) for i in bad], values=pw[bad].astype(w.dtype))

# R2: parts whose weight norm left the parent's by more than 8 % (outside 0.93-1.08, the healthy range is 0.98-1.07): rescale to the parent's norm
cand = []
for key, names in blocks().items():
    own = np.sqrt(sum(norm(T()[n]) ** 2 for n in names)); par = np.sqrt(sum(norm(pt[n]) ** 2 for n in names))
    if own > 0 and not (0.93 <= own / par <= 1.08):
        cand.append((abs(np.log(own / par)), key, names, float(par / own)))
cand.sort(reverse=True)
for _, key, names, f in cand[:3]:
    try_edit(f"R2 part {key} has norm {1 / f:.2f} x the parent's: rescale", "CER falls", f"part {key} scale", "scale", names, factor=f)

if True:
    # R3: a part whose tensors are all zero: transplant the parent's
    for key, names in blocks().items():
        if s.budget_left < 3:
            break
        if all(norm(T()[n]) == 0 for n in names if n.endswith("weight")):
            try_edit(f"R3 part {key} is all zero: transplant the parent's", "CER falls", f"part {key} from parent", "transplant", names,
                     per_tensor={n: {"donor": pt[n]} for n in names})
    # R4: a weight matrix far outside the healthy relative distance (> 0.30; healthy maximum 0.24): transplant its whole part from the parent
    far = []
    for key, names in blocks().items():
        for n in names:
            if n.endswith(".weight") and T()[n].ndim == 2 and "norm" not in n:
                d = norm(T()[n] - pt[n]) / norm(pt[n])
                if d > 0.30:
                    far.append((d, key, names))
                break
    far.sort(reverse=True)
    for d, key, names in far[:2]:
        try_edit(f"R4 part {key} is {d:.2f} from the parent (healthy maximum 0.24): transplant the parent's part", "CER falls", f"part {key} from parent",
                 "transplant", names, per_tensor={n: {"donor": pt[n]} for n in names})
    # R5: an attention head (64 columns of an output projection) that is all zero: copy the parent's columns
    for key, names in blocks().items():
        if s.budget_left < 2:
            break
        n = f"{key[0]}.{key[1]}.{key[2]}.o.weight"
        if n in T() and key[2] in ("self", "cross", "attn"):
            w = T()[n]
            zero_cols = [j for j in range(0, w.shape[1], 64) if not w[:, j:j + 64].any()]
            if zero_cols:
                idx = [c for j in zero_cols for c in range(j, j + 64)]
                try_edit(f"R5 {n}: {len(zero_cols)} head(s) have all-zero columns: copy the parent's columns", "CER falls", "head from parent",
                         "set_cols", [n], index=idx, values=pt[n][:, idx].astype(w.dtype))
    # R6: a layer-norm bias or weight vector far from the parent's (relative distance > 0.5; healthy maximum 0.13): copy the parent's vector
    badn = [n for n in T() if "norm" in n and n in pt and norm(pt[n]) > 0 and norm(T()[n] - pt[n]) / norm(pt[n]) > 0.5]
    if badn:
        try_edit(f"R6 {len(badn)} layer-norm vectors are > 0.5 from the parent: copy the parent's", "CER falls", "norm vectors from parent",
                 "transplant", badn, per_tensor={n: {"donor": pt[n]} for n in badn})
s.freeze()
