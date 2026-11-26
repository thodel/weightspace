import numpy as np
from wsp.run import open_session


def deletion_profile(conf):
    """Summarize deletions in the most frequent aligned confusion pairs."""
    rows = []
    for entry in (conf.items() if isinstance(conf, dict) else conf):
        if isinstance(entry, dict):
            r, p, n = entry.get('ref'), entry.get('hyp'), entry.get('count', 0)
        elif isinstance(entry, (list, tuple)) and len(entry) == 3:
            r, p, n = entry
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            pair, n = entry
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            r, p = pair
        else:
            continue
        if isinstance(n, (int, float, np.number)):
            rows.append((int(n), r, p))
    rows.sort(key=lambda x: -x[0])
    dels = [(n, r) for n, r, p in rows[:10] if p in ('∅', '', None, '<eps>')]
    return len(dels), any(r == ' ' for _, r in dels)


def main():
    s = open_session('.')
    best = s.measure('C0')

    def decide(label):
        nonlocal best
        m = s.measure(label)
        old_profile = deletion_profile(best.confusions)
        new_profile = deletion_profile(m.confusions)
        reason = (f'{label}: CER {best.cer:.5f}->{m.cer:.5f}; collapsed '
                  f'{best.collapsed}->{m.collapsed}; top deletion profile '
                  f'(types, includes space) {old_profile}->{new_profile}')
        if m.collapsed > best.collapsed + 1 and m.cer < best.cer:
            s.revert(reason + '; reject collapse regression')
            return False
        ok = s.keep(reason)
        if ok:
            best = m
        return ok

    T = s.subject.tensors()
    names = sorted(n for n in T if n.startswith('L_16.layer.') and 'bias_ih' not in n)
    def bias_norm(layer):
        return np.sqrt(sum(float(np.sum(T[n].astype(np.float64)**2))
                           for n in T if n.startswith(layer + '.layer.bias_hh')))
    estimate = float(np.clip(bias_norm('L_14') / max(bias_norm('L_16'), 1.e-9), 1.5, 3.0))
    gain = 1.0
    tried = []
    for target in (estimate, 2.0, 2.5, 5.0/3.0, 3.0):
        if any(abs(target - x) < 0.025 for x in tried):
            continue
        tried.append(target)
        h = s.hypothesis(f'L_16 gates and candidate updates have attenuated preactivations; test original-state gain {target:.4f}.',
                         'CER down 5-15 pp from C0 if repaired, or 0.1-2 pp versus a nearby retained gain.',
                         'joint L_16 scale')
        s.edit('scale', names, hypothesis=h, factor=target/gain)
        if decide(f'L16 x{target:.4f}'):
            gain = target

    b = 'O_18.lin.bias'
    def shift(row, amount, label, pred):
        d = np.zeros_like(s.subject.tensors()[b])
        d[row] = amount
        h = s.hypothesis(label + ': a class prior contributes to residual deletions.', pred, 'single output bias add')
        s.edit('add', [b], hypothesis=h, delta=d)
        return decide(label)

    if deletion_profile(best.confusions)[0] >= 3:
        kept = shift(0, -0.75, 'blank offset -0.75', 'CER down 0.2-2 pp')
        if deletion_profile(best.confusions)[0] >= 3:
            target = -1.5 if kept else -0.3
            shift(0, target - (-0.75 if kept else 0), f'blank offset {target}', 'CER down 0.2-2 pp')
    cm = s.subject.codec_map()
    if deletion_profile(best.confusions)[1] and ' ' in cm:
        shift(cm[' '], 0.6, 'space offset +0.6', 'CER down 0.1-0.8 pp')
    s.freeze()


if __name__ == '__main__':
    main()
