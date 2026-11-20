import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'
W = 'O_18.lin.weight'


def reported_e_errors(confusions):
    """Count reported edits involving e (a diagnostic, not a full error rate)."""
    total = 0
    rows = confusions.items() if isinstance(confusions, dict) else confusions
    for item in rows:
        if isinstance(item, dict):
            a, b = item.get('ref'), item.get('hyp')
            n = item.get('count', item.get('n', 0))
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            a, b, n = item
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            pair, n = item
            if isinstance(pair, str):
                pair = pair.replace('->', '→').split('→')
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            a, b = pair
        else:
            continue
        if 'e' in (str(a).strip(), str(b).strip()):
            total += int(n)
    return total


def main():
    s = open_session('.')
    best = s.measure('C0')
    cm = s.subject.codec_map()
    parent = KrakenSubject(PARENT)
    pm = parent.codec_map()
    i = cm['e']
    wp = parent.tensors()[W]
    donor = wp[pm['e']].copy()
    wc = s.subject.tensors()[W]
    letters = [c for c in 'abcdfghijklmnopqrstuvwxyz' if c in cm and c in pm]
    offset = np.median([wc[cm[c]] - wp[pm[c]] for c in letters], axis=0)
    c0_errors = reported_e_errors(best.confusions)

    def attempt(label, why, prediction, row):
        nonlocal best
        h = s.hypothesis(why, prediction, 'set_rows: lowercase e weight only')
        s.edit('set_rows', [W], hypothesis=h, index=[i], values=row[None, :])
        m = s.measure(label)
        err = reported_e_errors(m.confusions)
        summary = (f'{label}: CER {best.cer:.5f} -> {m.cer:.5f}; collapsed '
                   f'{best.collapsed} -> {m.collapsed}; reported e-involving errors '
                   f'{c0_errors} at C0, {err} now')
        # A row repair should not harm whole lines or leave the original
        # systematic e failure intact. Always require the paired keep rule.
        if (m.collapsed > best.collapsed or (c0_errors and err >= c0_errors)) and m.cer < best.cer:
            s.revert(summary + '; failed local diagnostic')
            return False
        if s.keep(summary):
            best = m
            return True
        return False

    first = attempt('H1-parent-e', 'Restore the erased e detector while preserving the child bias.',
                    'CER down 9-12 percentage points from C0.', donor)
    second = attempt('H2-lowercase-drift', 'Other lowercase rows estimate shared feature drift.',
                     'CER down 0.05-0.5 pp vs H1.' if first else 'CER down 9-12 pp vs C0.', donor + offset)
    if first or second:
        anchor = s.subject.tensors()[W][i].copy()
        up = attempt('H3-gain-1.10', 'Test stronger e evidence against adapted competitors.',
                     'CER down 0.05-0.6 pp if donor evidence is weak.', 1.10 * anchor)
        gain = 1.20 if up else 0.90
        attempt(f'H3-gain-{gain:.2f}', 'Extend a successful gain, or test attenuation if boost failed.',
                'CER down 0.05-0.6 pp if this corrects the residual imbalance.', gain * anchor)
    s.freeze()


if __name__ == '__main__':
    main()
