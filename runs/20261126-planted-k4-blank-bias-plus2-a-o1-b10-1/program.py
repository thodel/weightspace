import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'
OUT = 'O_18.lin.bias'
CONV = 'C_3.co.bias'


def top_pairs(metrics):
    """Normalise ranked triples or a counter of (reference, hypothesis) pairs."""
    c = metrics.confusions
    entries = c.items() if isinstance(c, dict) else c
    pairs = []
    for x in entries:
        if isinstance(x, dict):
            r = x.get('ref', x.get('reference'))
            p = x.get('hyp', x.get('hypothesis'))
            n = x.get('count', x.get('n', 0))
            if r is None and p is None:
                continue
        elif isinstance(x, (tuple, list)) and len(x) == 3:
            r, p, n = x
        elif isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[0], (tuple, list)) and len(x[0]) == 2:
            (r, p), n = x
        else:
            continue
        if isinstance(n, (float, int, np.number)):
            pairs.append((r, p, int(n)))
    return sorted(pairs, key=lambda t: -t[2])[:10]


def main():
    s = open_session('.')
    best = s.measure('C0')
    space = s.subject.codec_map()[' ']
    gaps = (None, '', '∅', '<del>', '<eps>')

    def persists(space_only=False):
        rows = top_pairs(best)
        if not rows:
            # C0 pack provides the initial counts; otherwise don't infer a
            # continuing symptom from a confusion format we cannot interpret.
            return best is baseline
        return any(p in gaps and n >= (15 if space_only else 20) and
                   (not space_only or r == ' ') for r, p, n in rows)

    def trial(label, text, prediction, op, name, **args):
        nonlocal best
        if not s.budget_left:
            return False
        h = s.hypothesis(text, prediction, op)
        s.edit(op, [name], hypothesis=h, **args)
        m = s.measure(label)
        s.note(f'{label}: collapsed {best.collapsed} -> {m.collapsed}; leading confusions: {str(m.confusions)[:320]}')
        if not (m.cer < best.cer and m.collapsed <= best.collapsed):
            s.revert('No CER gain or collapse count rose')
            return False
        if s.keep('CER lower, collapse count did not rise; require paired support'):
            best = m
            return True
        return False

    def shift(row, amount, label, prediction):
        d = np.zeros_like(s.subject.tensors()[OUT])
        d[row] = amount
        desc = 'CTC blank suppresses genuine emissions' if row == 0 else 'Literal spaces lose at boundaries'
        return trial(label, f'{desc}; change bias by {amount:+g}', prediction, 'add', OUT, delta=d)

    baseline = best
    if shift(0, -2, 'blank-2', 'CER down 0.4-2 pp'):
        if persists() and shift(0, -2, 'blank-extension', 'CER down 0.1-0.8 pp') and persists():
            shift(0, -1, 'blank-final', 'CER down 0.1-0.8 pp')
    else:
        shift(0, -1, 'blank-half', 'CER down 0.1-0.6 pp')

    parent = KrakenSubject(PARENT)
    donor = parent.tensors()[CONV].copy()
    residual = s.subject.tensors()[CONV] - donor
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    extremes = np.abs(residual - med) > max(0.002, 5 * 1.4826 * mad)
    correction = np.zeros_like(residual)
    if abs(med) > max(0.001, 2 * mad):
        correction.fill(-med)
    elif 0 < extremes.sum() <= len(residual) // 4:
        correction[extremes] = -residual[extremes]
    if not (np.any(correction) and trial('C3-selective', 'Remove structured C3 bias drift while retaining other adaptations', 'CER down 0.1-0.8 pp', 'add', CONV, delta=correction)):
        trial('C3-donor', 'C3 bias displacement weakens early detection', 'CER down 0.1-0.7 pp', 'transplant', CONV, donor=donor)

    if persists(space_only=True):
        if shift(space, 2, 'space+2', 'CER down 0.2-1 pp'):
            if persists(space_only=True):
                shift(space, 2, 'space-extension', 'CER down 0.1-0.5 pp')
        else:
            shift(space, 1, 'space-half', 'CER down 0.1-0.4 pp')
    s.freeze()


if __name__ == '__main__':
    main()
