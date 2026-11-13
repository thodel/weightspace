"""Cross-lineage exchange (RQ3 across lineages, docs/MODEL_FAMILY.md "Next" item 2): align the hidden units of
two unrelated group-A kraken models, then exchange blocks in the aligned coordinates.

Between independently trained networks hidden units are in arbitrary order, so an element-wise block exchange
is meaningless (``wsp.exchange`` refuses it). A permutation of the units of a layer, applied consistently to the
layer's outputs and to every weight that reads them, leaves the network's function unchanged. So:

1. **match** — read the same probe lines with both models; per layer, correlate every unit of model A with every
   unit of model B (conv: a channel over all its (height, frame) positions; LSTM: a hidden unit per direction over
   all frames), and solve the assignment that maximises the summed correlation (Hungarian,
   ``scipy.optimize.linear_sum_assignment``);
2. **permute** — reorder B's units to A's order: conv output channels (weight rows, bias) and the next layer's
   input channels; for ``C_9`` the inputs of ``L_12``, whose 960 features are ``h * 64 + c`` (kraken's ``S_11``
   reshape, checked numerically); for an LSTM per direction the rows of each gate block ``[i, f, g, o]`` of
   ``weight_ih``, ``weight_hh``, ``bias_ih``, ``bias_hh``, the columns of ``weight_hh``, and the next layer's
   input columns ``[forward, 200 + backward]``; the output layer's columns;
3. **verify** — the permuted B must read every line exactly as B does (same text), else the permutation is wrong;
4. **exchange** — B's (aligned) block replaces A's, or A moves half-way to it; optionally each transplanted conv
   channel is rescaled to A's channel's activation scale first (ReLU is positively homogeneous, so scaling a
   channel's weights and bias by a > 0 scales its output by a exactly).

Group A only: ``Cr3,13,32 Mp Cr3,13,32 Mp Cr3,9,64 Mp Cr3,9,64 S Lbx200 x3 O`` (docs/MODEL_FAMILY.md).
"""
from __future__ import annotations

import numpy as np

CONV = ["C_0", "C_3", "C_6", "C_9"]
LSTM = ["L_12", "L_14", "L_16"]
H9 = 15        # height after the three max-pools (120 / 8)
HID = 200


def _w(b):
    return f"{b}.co.weight", f"{b}.co.bias"


def _l(b, rev=False):
    s = "_reverse" if rev else ""
    return [f"{b}.layer.{k}_l0{s}" for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]


def unit_activations(subject, items, blocks=CONV + LSTM) -> dict[str, np.ndarray]:
    """Per block a (samples x units) matrix: conv channels over every (height, frame) position, LSTM units
    (400 = forward 200 + backward 200) over every frame. Lines are concatenated in order."""
    acts = subject.activations(items, list(blocks))
    out = {}
    for b in blocks:
        X = np.concatenate(acts[b], axis=0)                 # frames x features
        if b.startswith("C_"):
            C = subject.tensors()[_w(b)[0]].shape[0]
            H = X.shape[1] // C
            # activations() flattens a conv output (C, H, W) as C*H rows per frame, channel-major
            X = X.reshape(-1, C, H).transpose(0, 2, 1).reshape(-1, C)   # (frames*H) x C
        out[b] = X.astype(np.float64)
    return out


def correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of A with every column of B (a constant column correlates 0)."""
    A = A - A.mean(0); B = B - B.mean(0)
    sa = np.sqrt((A ** 2).sum(0)); sb = np.sqrt((B ** 2).sum(0))
    C = (A.T @ B) / np.outer(np.where(sa > 0, sa, 1), np.where(sb > 0, sb, 1))
    C[sa == 0, :] = 0; C[:, sb == 0] = 0
    return C


def match(actsA: dict, actsB: dict, max_samples: int = 200000, seed: int = 20261108) -> dict:
    """Per layer the permutation ``p`` with B-unit ``p[i]`` matched to A-unit ``i``, plus statistics."""
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(seed)
    res = {}
    for b in actsA:
        XA, XB = actsA[b], actsB[b]
        if XA.shape[0] != XB.shape[0]:
            raise ValueError(f"{b}: {XA.shape[0]} vs {XB.shape[0]} samples; the same lines must be read")
        if XA.shape[0] > max_samples:
            idx = rng.choice(XA.shape[0], max_samples, replace=False); XA, XB = XA[idx], XB[idx]
        if b.startswith("L_"):          # match within each direction
            perm = np.empty(2 * HID, dtype=int); corr = np.empty(2 * HID)
            for d in (0, 1):
                sl = slice(d * HID, (d + 1) * HID)
                C = correlation(XA[:, sl], XB[:, sl])
                r, c = linear_sum_assignment(-C)
                perm[sl] = c + d * HID; corr[sl] = C[r, c]
        else:
            C = correlation(XA, XB)
            r, c = linear_sum_assignment(-C)
            perm, corr = c, C[r, c]
        deadA = (XA.std(0) == 0); deadB = (XB.std(0) == 0)
        res[b] = {"perm": perm, "corr": corr, "dead_A": deadA, "dead_B": deadB[perm],
                  "std_A": XA.std(0), "std_B": XB.std(0)[perm],
                  "mean_corr": float(corr.mean()), "median_corr": float(np.median(corr)),
                  "identity_mean_corr": float(np.mean([correlation(XA[:, [i]], XB[:, [i]])[0, 0] for i in range(XA.shape[1])]))}
    return res


def permute_tensors(t: dict[str, np.ndarray], perms: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """B's tensors with the units of every block in ``perms`` reordered (new unit i = old unit perm[i]).
    Every weight that reads a permuted layer is permuted on its input side."""
    t = {k: v.copy() for k, v in t.items()}
    order = CONV + LSTM
    for k, b in enumerate(order):
        if b not in perms:
            continue
        p = np.asarray(perms[b])
        nxt = order[k + 1] if k + 1 < len(order) else "O_18"
        if b.startswith("C_"):
            w, bias = _w(b)
            t[w] = t[w][p]; t[bias] = t[bias][p]
            if nxt.startswith("C_"):
                t[_w(nxt)[0]] = t[_w(nxt)[0]][:, p]
            else:                                   # C_9 -> S_11 -> L_12: feature j = h * C + c
                C = len(p)
                cols = np.array([h * C + p[c] for h in range(H9) for c in range(C)])
                for rev in (False, True):
                    wi = _l(nxt, rev)[0]
                    t[wi] = t[wi][:, cols]
        else:
            for d, rev in ((0, False), (1, True)):
                pd = p[d * HID:(d + 1) * HID] - d * HID
                rows = np.concatenate([g * HID + pd for g in range(4)])
                wi, wh, bi, bh = _l(b, rev)
                t[wi] = t[wi][rows]; t[wh] = t[wh][rows][:, pd]; t[bi] = t[bi][rows]; t[bh] = t[bh][rows]
            if nxt.startswith("L_"):
                for rev in (False, True):
                    wi = _l(nxt, rev)[0]
                    t[wi] = t[wi][:, p]
            else:
                t["O_18.lin.weight"] = t["O_18.lin.weight"][:, p]
    return t


def block_tensors(b: str) -> list[str]:
    return list(_w(b)) if b.startswith("C_") else _l(b, False) + _l(b, True)


def rescale_conv(tB: dict, b: str, scale: np.ndarray) -> dict:
    """Multiply conv block b's output channel c by scale[c] (> 0): weights and bias of the channel."""
    t = dict(tB)
    w, bias = _w(b)
    t[w] = tB[w] * scale[:, None, None, None]; t[bias] = tB[bias] * scale
    return t
