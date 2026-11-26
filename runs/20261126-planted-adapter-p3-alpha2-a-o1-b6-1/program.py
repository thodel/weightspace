import numpy as np
from wsp.run import open_session


def main():
    s = open_session('.')
    b_names = sorted(n for n in s.subject.tensors() if n.endswith('.lora_B.weight'))
    c0 = s.measure('C0')
    if len(b_names) != 128 or s.subject.scale != 4.0:
        s.note('Adapter does not match the preregistered module count and gain; no edit.')
    else:
        trials = [
            ('H1', 0.5, 'The doubled alpha overdrives every module; restore effective alpha 128.',
             'CER down 1.3-2.3 absolute units vs C0; 15-28 fewer pages at cap.'),
            ('H2', 0.56, 'Match the average sibling total update norm rather than alpha.',
             'CER down 1.0-2.2 absolute units vs C0; 10-25 fewer pages at cap.'),
        ]
        # A rejected trial is undone before the next one: both factors are relative to C0.
        for label, factor, claim, prediction in trials:
            if s.budget_left < 1:
                break
            h = s.hypothesis(claim, prediction, 'global B scale')
            s.edit('scale', b_names, hypothesis=h, factor=factor)
            m = s.measure(label)
            if not (np.isfinite(m.cer) and m.cer < c0.cer):
                s.revert('No finite CER improvement versus C0.')
                continue
            if s.keep('Lower measured CER; require paired improvement and no increase in capped pages.'):
                break
    s.freeze()


if __name__ == '__main__':
    main()
