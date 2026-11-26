import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'
W, B, CB = 'O_18.lin.weight', 'O_18.lin.bias', 'C_3.co.bias'


def peak(conf):
    # Largest reported confusion count (Counter, counted pairs or triples).
    vals = []
    for item in (conf.items() if isinstance(conf, dict) else conf):
        if isinstance(item, dict):
            n = item.get('count', item.get('n', 0))
        elif isinstance(item, (list, tuple)) and item:
            n = item[-1]
        else:
            continue
        if isinstance(n, (int, float, np.number)):
            vals.append(int(n))
    return max(vals, default=0)


def main():
    s = open_session('.')
    best = s.measure('C0')
    remaining = min(9, s.budget_left)

    def trial(label, rationale, prediction, op, names, **kwargs):
        nonlocal best, remaining
        if remaining <= 0:
            return False, False
        h = s.hypothesis(rationale, prediction, op)
        s.edit(op, names, hypothesis=h, **kwargs)
        m = s.measure(label)
        remaining -= 1
        delta = m.cer - best.cer
        p0, p1 = peak(best.confusions), peak(m.confusions)
        reason = (f'{label}: CER change {delta:+.6f}; collapsed {best.collapsed}->{m.collapsed}; '
                  f'largest reported confusion {p0}->{p1}')
        if delta >= 0 or m.collapsed > best.collapsed or p1 > max(p0 + 16, 1.4*p0):
            s.revert(reason)
            kept = False
        else:
            kept = s.keep(reason)
            if kept:
                best = m
        return kept, delta

    logit_unit = 1.
    kept, _ = trial('H1', 'Remove a common twofold output gain without moving argmax boundaries.',
                    'Near zero; 0.1-0.5 pp lower CER only if confidence affects decoding.',
                    'scale', [W, B], factor=0.5)
    if kept:
        logit_unit = 0.5
    else:
        kept, _ = trial('H2-W', 'Only the feature logits may be amplified relative to the learned priors.',
                        '0.1-0.8 pp lower if feature gain is excessive.', 'scale', [W], factor=0.5)
        if kept:
            logit_unit = 0.5
        else:
            trial('H2-b', 'Test the opposite imbalance: bias magnitudes too large relative to features.',
                  '0.1-0.8 pp lower if prior gain is excessive.', 'scale', [B], factor=0.5)

    donor = KrakenSubject(PARENT).tensors()[CB].copy()
    kept, d = trial('H3', 'Restore donor C_3 channel thresholds while preserving the adapted kernels.',
                    '0.05-0.4 pp lower CER.', 'interpolate', [CB], other=donor, alpha=1.)
    if not kept and d < 0:
        trial('H3-partial', 'Partial threshold repair might preserve useful adaptation.',
              '0.05-0.2 pp lower CER.', 'interpolate', [CB], other=donor, alpha=0.5)

    def shift(label, row, dose, why, pred):
        d = np.zeros_like(s.subject.tensors()[B])
        d[row] = dose * logit_unit
        return trial(label, why, pred, 'add', [B], delta=d)

    kept, d = shift('H4', 0, -1., 'Reduce blank competition to recover deleted symbols.', '0.1-0.6 pp lower CER.')
    if kept:
        shift('H4-more', 0, -1., 'Test residual over-blanking.', 'Up to 0.2 pp further reduction.')
    elif d <= 0:
        shift('H4-partial', 0, -0.5, 'Reduce blank less to limit insertions.', 'Up to 0.2 pp lower CER.')

    row = s.subject.codec_map().get(' ')
    if row is not None and remaining:
        kept, _ = shift('H5', row, 1., 'Promote word breaks; omissions outnumber spurious spaces at C0.',
                        '0.05-0.3 pp lower CER.')
        if kept and remaining:
            shift('H5-more', row, 1., 'Check for remaining space threshold error.', '0.02-0.15 pp further reduction.')
    s.freeze()


if __name__ == '__main__':
    main()
