import numpy as np

from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session


PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'


def power(x):
    return float(np.mean(np.square(x, dtype=np.float64)))


def centered(x):
    return x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + x.mean()


def denoise(x, refs):
    # x and refs are fine-tuning deltas, not full weights. Check for a
    # substantial excess over both adjacent layers before touching this gate.
    signal = float(np.mean([power(z) for z in refs]))
    total = power(x)
    if total < 2.4 * signal or total < 1e-12:
        return x
    a = centered(x)
    n, m = a.shape
    sig2 = float(np.mean([power(centered(z)) for z in refs]))
    bulk = power(a)
    if bulk < 2.0 * sig2:
        # Not the predicted diffuse noise signature.
        return x
    noise2 = bulk - sig2
    # Keep the DC, row and column components: averaging over hundreds of
    # independent perturbations already makes those estimates much cleaner.
    means = x - a
    u, d, vh = np.linalg.svd(a, full_matrices=False)
    # Separate coherent fine-tuning patterns from the nearly isotropic
    # residual. Rectangular additive-noise singular-value shrinker, treating
    # even unstructured signal as part of the background for this step.
    edge = np.sqrt(bulk) * (np.sqrt(n) + np.sqrt(m))
    strong = d > edge
    spikes = np.zeros_like(d)
    t = d[strong] ** 2 - (n + m) * bulk
    spikes[strong] = np.sqrt(np.maximum(t*t - 4*n*m*bulk**2, 0.0)) / d[strong]
    structured = (u * spikes) @ vh
    # A trained update also has a full-rank tail. Don't zero it: estimate
    # the signal fraction left after accounting for the coherent component.
    tail_sig2 = max(0.10 * sig2, sig2 - power(structured))
    gain = tail_sig2 / (tail_sig2 + noise2)
    return means + structured + gain * (a - structured)


def main():
    s = open_session('.')
    parent = KrakenSubject(PARENT)
    p = parent.tensors()
    c = s.subject.tensors()
    args = {}
    for kind in ('ih', 'hh'):
        for direction in ('', '_reverse'):
            suffix = f'.layer.weight_{kind}_l0{direction}'
            key = 'L_14' + suffix
            def delta(layer):
                name = f'L_{layer}' + suffix
                return c[name].astype(np.float64) - p[name]
            original = delta(14)
            neighbors = [delta(layer) for layer in (12, 16)]
            result = original.copy()
            for rows in np.split(np.arange(original.shape[0]), 4):
                result[rows] = denoise(original[rows], [z[rows] for z in neighbors])
            if not np.array_equal(result, original):
                args[key] = {'delta': (result - original).astype(c[key].dtype)}
    if args:
        h = s.hypothesis('L_14 input and recurrent matrices contain excess diffuse noise in both directions, unlike the adjoining layers.',
                         'Joint repair: CER lower by about 2-5 percentage points, with fewer deletions.',
                         'parent-relative matrix denoising')
        s.edit('add', list(args), per_tensor=args, hypothesis=h)
        s.keep_unmeasured('Remove the unusually large diffuse component without replacing the entire trained middle layer.')
    s.freeze()


if __name__ == '__main__':
    main()
