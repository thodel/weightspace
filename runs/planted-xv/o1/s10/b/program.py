import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'


def correct_slices(w, p, axis):
    # Each unit along axis is an entire kernel/filter bank or gate vector.
    x = np.moveaxis(w, axis, 0)
    y = np.moveaxis(p, axis, 0)
    a = x.reshape(len(x), -1).astype(np.float64)
    b = y.reshape(len(y), -1).astype(np.float64)
    nx = np.linalg.norm(a, axis=1)
    ny = np.linalg.norm(b, axis=1)
    if len(x) < 8 or a.shape[1] < 24:
        return 0
    valid = ny > max(1.e-7, 0.05 * np.median(ny))
    gain = nx / np.maximum(ny, 1.e-12)
    cosine = np.sum(a*b, axis=1) / np.maximum(nx*ny, 1.e-16)
    peers = valid & (cosine > 0.8)
    if np.sum(peers) < 0.75 * len(x):
        return 0
    usual = np.median(gain[peers])
    n = 0
    for i in np.flatnonzero(valid):
        if nx[i] == 0:
            x[i] = y[i]
        elif cosine[i] < -0.88 or (cosine[i] > 0.91 and
              (gain[i] < usual / 3.5 or gain[i] > usual * 3.5)):
            factor = np.sign(cosine[i])
            if gain[i] < usual / 2 or gain[i] > usual * 2:
                factor *= usual / gain[i]
            x[i] *= factor
        else:
            continue
        n += 1
    return n


def main():
    s = open_session('.')
    parent = KrakenSubject(PARENT)
    t = s.subject.tensors()
    pt = parent.tensors()
    key = 'C_3.co.bias'
    x, p = t[key], pt[key]
    residual = x.astype(np.float64) - p
    center = np.median(residual)
    sigma = 1.4826 * np.median(np.abs(residual-center))
    rms = np.sqrt(np.mean(p.astype(np.float64)**2))
    shift = center if abs(center) > max(2*sigma, 0.3*rms) else 0.
    spikes = np.abs(residual-center) > max(7*sigma, 1.5*rms, 1.e-5)
    if shift or spikes.any():
        target = x.copy() - np.float32(shift)
        target[spikes] = p[spikes] + (center-shift)
        h = s.hypothesis('Abnormal channel thresholds in C_3 suppress visual evidence.',
                         'If causal, CER down 0.1-0.4 absolute and fewer collapses.', 'bias correction')
        s.edit('add', [key], hypothesis=h, delta=(target-x).astype(x.dtype))
        s.keep_unmeasured('Remove only a resolved shared offset or exceptional threshold residuals.')

    for key in sorted(t):
        if 'weight' not in key or not key.startswith(('C_', 'L_')):
            continue
        x, p = t[key], pt[key]
        w = x.copy()
        changed = sum(correct_slices(w, p, axis) for axis in (0, 1))
        if changed:
            h = s.hypothesis('An erased, reversed or extreme-gain channel in '+key+' impairs recognition.',
                             'If causal, CER down roughly 0.05-0.3 absolute.', 'localized channel correction')
            s.edit('add', [key], hypothesis=h, delta=w-x)
            s.keep_unmeasured('Repair structurally anomalous slices only; preserve ordinary adaptation.')
    s.freeze()


if __name__ == '__main__':
    main()
