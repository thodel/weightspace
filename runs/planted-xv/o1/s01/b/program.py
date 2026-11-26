import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

DONOR = '/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel'


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def match(scores):
    """One-to-one row -> column assignment by a finite-epsilon auction."""
    n = scores.shape[0]
    price = np.zeros(n)
    owner = np.full(n, -1, dtype=int)
    assignment = np.empty(n, dtype=int)
    pending = list(range(n))
    while pending:
        i = pending.pop()
        v = scores[i] - price
        j = int(v.argmax())
        best = v[j]
        v[j] = -np.inf
        price[j] += best - v.max() + 0.001
        if owner[j] >= 0:
            pending.append(int(owner[j]))
        assignment[i], owner[j] = j, i
    return assignment


def consumers(tensors):
    # For each C_3 channel: Gram of the 27 spatial taps of C_6 across
    # its 64 output filters. This does not assume aligned C_6 outputs.
    w = tensors['C_6.co.weight'].astype(np.float64).transpose(1, 0, 2, 3)
    w = w.reshape(w.shape[0], w.shape[1], -1)
    g = (w.transpose(0, 2, 1) @ w).reshape(w.shape[0], -1)
    g = unit(g)
    return unit(g - g.mean(axis=0))


def main():
    s = open_session('.')
    t = s.subject.tensors()
    wk, bk = 'C_3.co.weight', 'C_3.co.bias'
    if not np.any(t[wk]) and not np.any(t[bk]):
        donor = KrakenSubject(DONOR)
        d = donor.tensors()
        key = 'C_0.co.weight'
        a = d[key].reshape(d[key].shape[0], -1)
        b = t[key].reshape(t[key].shape[0], -1)
        # For each donor input find a target channel of the same polarity;
        # inversions cannot be freely absorbed across a ReLU.
        p = match(unit(a) @ unit(b).T)
        q = match(consumers(t) @ consumers(d).T)
        gain = np.clip(np.linalg.norm(a, axis=1) / np.maximum(np.linalg.norm(b[p], axis=1), 1e-8), 0.5, 2.)
        w = np.empty_like(t[wk])
        w[:, p] = d[wk][q] * gain[None, :, None, None]
        h = s.hypothesis('Zero C_3 cuts off all image evidence. Bridge it with channel-matched donor stroke filters.',
                         'Fewer empty lines; CER down roughly 5-25 percentage points.',
                         'channel-aligned convolution transplant')
        s.edit('transplant', [wk, bk], hypothesis=h,
               per_tensor={wk: {'donor': w}, bk: {'donor': d[bk][q]}})
        s.keep_unmeasured('Restore an image-dependent path without overwriting any nonzero subject tensor.')
    s.freeze()


if __name__ == '__main__':
    main()
