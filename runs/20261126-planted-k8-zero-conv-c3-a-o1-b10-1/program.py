from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'


def main():
    s = open_session('.')
    best = s.measure('C0')
    def names(layer):
        return [layer + '.co.' + suffix for suffix in ('weight', 'bias')]

    broken = names('C_3')
    if any(np.any(s.subject.tensors()[k]) for k in broken):
        s.note('C_3 is not erased as expected; leave it untouched.')
        s.freeze()
        return

    parent = KrakenSubject(PARENT)
    donor = parent.tensors()
    h = s.hypothesis('Erased C_3 prevents transmission of image features; restore the parental filters and offsets.',
                     'CER down 80-98 percentage points; fewer empty outputs.', 'block transplant')
    s.edit('transplant', broken, hypothesis=h,
           per_tensor={k: {'donor': donor[k]} for k in broken})
    m = s.measure('C_3 restored')
    if m.cer < best.cer and m.collapsed < best.collapsed and m.confusions != best.confusions:
        restored = s.keep('Collapse eased, CER fell, and confusions changed; test paired improvement.')
    else:
        s.revert('No corroborated recovery from blank collapse.')
        restored = False
    if restored:
        best = m
        for layer, alpha, why in [('C_0', 0.5, 'Better match the input channels expected by donor C_3.'),
                                  ('C_6', 0.25, 'Better match the feature channels produced by donor C_3.')]:
            if best.cer <= 0.03:
                break
            keys = names(layer)
            h = s.hypothesis(why, 'CER down 0.1-2 percentage points.', 'interface interpolation')
            s.edit('interpolate', keys, hypothesis=h, alpha=alpha,
                   per_tensor={k: {'other': donor[k]} for k in keys})
            m = s.measure(layer + ' partial donor match')
            if m.cer < best.cer and m.collapsed <= best.collapsed:
                if s.keep('Lower CER without additional collapse; require paired evidence.'):
                    best = m
            else:
                s.revert('No CER gain or more collapsed readings.')
    s.freeze()


if __name__ == '__main__':
    main()
