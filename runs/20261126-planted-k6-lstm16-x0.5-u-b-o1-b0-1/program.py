import numpy as np
from wsp.run import open_session


def inverse_gain(tensors, names):
    # A rounded product by a non-power-of-two scalar skips some fp32
    # numbers. Check reachability, not just agreement of RMS statistics.
    samples = [tensors[n].reshape(-1)[::17] for n in names if 'weight' in n]
    matches = []
    for k in range(40, 61):
        if k == 50:  # a binary shift round-trips any normal number
            continue
        a = np.float32(k / 100.0)
        if all(np.all((x / a) * a == x) for x in samples):
            matches.append(float(a))
    return 1.0 / matches[0] if len(matches) == 1 else 2.0


def main():
    s = open_session('.')
    tensors = s.subject.tensors()
    names = ['L_16.layer.' + stem + suffix
             for suffix in ('', '_reverse')
             for stem in ('weight_ih_l0', 'weight_hh_l0', 'bias_hh_l0')]
    gain = inverse_gain(tensors, names)
    h = s.hypothesis('Both directions of the final BiLSTM have attenuated gate preactivations.',
                     'CER down 0.08-0.15 absolute with fewer deletions and collapsed lines.',
                     'scale L16 nonzero tensors')
    s.edit('scale', names, hypothesis=h, factor=gain)
    s.keep_unmeasured('The roughly twofold deficit spans input weights, recurrent weights and biases in both directions; restore their common scale without replacing features.')
    s.freeze()


if __name__ == '__main__':
    main()
