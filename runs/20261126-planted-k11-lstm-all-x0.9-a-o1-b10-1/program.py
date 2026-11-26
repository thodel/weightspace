import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'
OUT = 'O_18.lin.bias'


def pair_counts(m):
    """Recognize both Counter-style pairs and serialized confusion records."""
    obj = m.confusions
    result = {}
    for entry in (obj.items() if isinstance(obj, dict) else obj):
        if isinstance(entry, dict):
            a = entry.get('ref', entry.get('reference'))
            b = entry.get('hyp', entry.get('hypothesis'))
            n = entry.get('count', entry.get('n'))
            if a is None and b is None:
                continue
        elif isinstance(entry, (tuple, list)) and len(entry) == 3:
            a, b, n = entry
        elif isinstance(entry, (tuple, list)) and len(entry) == 2 and isinstance(entry[0], (tuple, list)) and len(entry[0]) == 2:
            (a, b), n = entry
        else:
            continue
        if isinstance(n, (int, float, np.number)):
            result[(str(a), str(b))] = int(n)
    return result


def sparse_lesion(w, p):
    delta = w - p
    repair = np.zeros_like(delta)
    for r, d in zip(np.split(repair, 4), np.split(delta, 4)):
        # Compare within one gate to avoid mistaking gate-to-gate drift for damage.
        rms_row = np.sqrt(np.mean(d*d, axis=1))
        rms_col = np.sqrt(np.mean(d*d, axis=0))
        rows = rms_row > 5 * max(1e-6, np.median(rms_row))
        cols = rms_col > 5 * max(1e-6, np.median(rms_col))
        mask = (rows[:, None] | cols[None, :] |
                (np.abs(d) > max(1.0, 30 * np.median(np.abs(d)))))
        if mask.mean() < .03:
            r[mask] = -d[mask]
    return repair


def main():
    s = open_session('.')
    current = s.measure('C0')
    donor = KrakenSubject(PARENT).tensors()

    def trial(label, text, prediction, name, op, **kwargs):
        nonlocal current
        h = s.hypothesis(text, prediction, op)
        s.edit(op, [name], hypothesis=h, **kwargs)
        m = s.measure(label)
        gain = current.cer - m.cer
        before, after = pair_counts(current), pair_counts(m)
        # Veto a large new concentrated confusion, not small shifts at the tail.
        spike = max((n - before.get(k, 0) for k, n in after.items()), default=0)
        reason = (f'{label}: CER gain {gain:.5f}; collapsed {current.collapsed} -> '
                  f'{m.collapsed}; largest reported pair increase {spike}')
        kept = False
        if m.collapsed > current.collapsed or spike > 50:
            s.revert(reason)
        else:
            kept = s.keep(reason)
        if kept:
            current = m
        return kept, gain, spike

    name = 'C_3.co.bias'
    trial('C3 bias only', 'Parent-like convolutional thresholds may recover weak features.',
          'CER down 0.02-0.25 pp', name, 'transplant', donor=donor[name])

    name = 'L_16.layer.weight_ih_l0'
    fix = sparse_lesion(s.subject.tensors()[name], donor[name])
    if np.any(fix):
        trial('localized L16 correction', 'Extreme gate-local donor residuals reflect a localized lesion.',
              'CER down 0.02-0.4 pp', name, 'add', delta=fix)
    else:
        s.note('No extreme localized L16 residual; skip H2.')

    space = s.subject.codec_map()[' ']

    def calibration(row, sign, title):
        applied = 0.
        for target in (1.5, 3.):
            delta = np.zeros_like(s.subject.tensors()[OUT])
            delta[row] = sign * (target - applied)
            label = f'{title} total {sign*target:+g}'
            if title == 'blank':
                pred = 'CER down 0.05-0.3 pp' if applied else 'CER down 0.1-0.5 pp'
                text = 'Blank competition suppresses letter emissions.'
            else:
                pred = 'CER down 0.03-0.2 pp' if applied else 'CER down 0.1-0.35 pp'
                text = 'Space under-emission contributes independently to deleted boundaries.'
            kept, gain, spike = trial(label, text, pred, OUT, 'add', delta=delta)
            if kept:
                applied = target
            elif gain < -.0006 or spike > 50:
                break
        return applied

    blank_change = calibration(0, -1, 'blank')
    space_change = calibration(space, 1, 'space')
    if not blank_change and not space_change:
        delta = np.zeros_like(s.subject.tensors()[OUT])
        delta[0], delta[space] = -1., 1.
        trial('joint mild', 'Marginal letters and boundaries need a joint, milder correction.',
              'CER down 0.1-0.4 pp', OUT, 'add', delta=delta)
    s.freeze()


if __name__ == '__main__':
    main()
