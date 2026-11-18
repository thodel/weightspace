import numpy as np
from wsp.run import open_session

s = open_session(".")
MASK = s.unseen_by()
T = s.subject.tensors()
LAYER = 8
MATS = ["self.q", "self.k", "self.v", "self.o", "cross.q", "cross.k", "cross.v", "cross.o", "ffn.in", "ffn.out"]


def nm(layer, k):
    return f"dec.{layer:02d}.{k}.weight"


def plan():
    """Per tensor: (name, observed norm, estimated clean norm, noise std)."""
    out = []
    for k in MATS:
        n = nm(LAYER, k)
        w = T[n]
        obs = float(np.linalg.norm(w))
        clean = 0.5 * (float(np.linalg.norm(T[nm(LAYER - 1, k)])) + float(np.linalg.norm(T[nm(LAYER + 1, k)])))
        if obs <= 1.06 * clean:
            continue
        noise = float(np.sqrt(max(obs ** 2 - clean ** 2, 0.0)))
        out.append((n, obs, clean, noise / np.sqrt(w.size)))
    return out


def svt_deltas(frac):
    d = {}
    for n, obs, clean, sigma in plan():
        w = T[n].astype(np.float64)
        u, sv, vt = np.linalg.svd(w, full_matrices=False)
        edge = frac * sigma * (np.sqrt(w.shape[0]) + np.sqrt(w.shape[1]))
        new = (u * np.maximum(sv - edge, 0.0)) @ vt
        d[n] = {"delta": (new - w).astype(np.float32)}
    return d


m0 = s.measure("C0", rule_mask=MASK)
base = m0.cer
P = plan()
s.note("dec.08 plan: " + str([(n, round(o, 1), round(c, 1), round(sg, 4)) for n, o, c, sg in P]))
names = [n for n, *_ in P]

kept_h1 = False
if names:
    h = s.hypothesis("dec.08 matrices carry additive Gaussian noise (excess norm over dec.07/dec.09 neighbours); Wiener shrinkage clean^2/obs^2",
                     "CER down 0.5 to 3 points", "scale")
    for n, o, c, sg in P:
        pass
    # one op per edit: use a common factor = mean of the per-tensor Wiener factors
    f = float(np.mean([c ** 2 / o ** 2 for n, o, c, sg in P]))
    s.edit("scale", names, hypothesis=h, factor=f)
    m1 = s.measure("H1 wiener scale", rule_mask=MASK)
    if m1.collapsed <= m0.collapsed + 2 and s.keep("wiener shrink of dec.08 matrices"):
        kept_h1 = True
        base = m1.cer
    else:
        if s._last_edit is not None:
            s.revert("H1 not kept")

    T = s.subject.tensors()
    d = svt_deltas(1.0)
    h = s.hypothesis("dec.08 noise has a flat spectrum; soft-threshold singular values at the Marchenko-Pastur edge removes noise, keeps signal",
                     "CER down 1 to 4 points, better than H1", "add (SVD soft threshold)")
    s.edit("add", list(d), hypothesis=h, per_tensor=d)
    m2 = s.measure("H2 svt edge", rule_mask=MASK)
    kept_h2 = False
    if m2.collapsed <= m0.collapsed + 2 and s.keep("svd soft threshold of dec.08 matrices"):
        kept_h2 = True
        base = m2.cer
    else:
        if s._last_edit is not None:
            s.revert("H2 not kept")

    if kept_h2:
        T = s.subject.tensors()
        s.note("H3 uses the plan on the already denoised state: norms are smaller, so plan() may skip tensors")
        # undo-free test: shrink further threshold only on tensors still above the 6% rule (usually none)
        d = svt_deltas(0.6)
        if d:
            h = s.hypothesis("H2 may over-shrink; a further gentle threshold", "within +-0.5 points; keep only if better", "add (SVD soft threshold)")
            s.edit("add", list(d), hypothesis=h, per_tensor=d)
            m3 = s.measure("H3 svt 0.6", rule_mask=MASK)
            if not s.keep("gentler extra threshold"):
                pass
else:
    s.note("no dec.08 inflation found; nothing applied")

summary = s.freeze()
