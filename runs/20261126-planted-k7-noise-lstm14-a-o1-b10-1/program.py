import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'


def tail_mask(values, mult, z):
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return values > max(mult * med, med + z * 1.4826 * mad, 1e-6)


def local_mask(w, p):
    # Compare like gates: normal magnitudes differ substantially across gates.
    d = np.abs(w - p)
    out = np.zeros_like(d, dtype=bool)
    for a, mask in zip(np.split(d, 4), np.split(out, 4)):
        mask[:] = tail_mask(a, 24, 18)
        for axis in (0, 1):
            norm = np.sqrt(np.mean(a*a, axis=axis))
            bad = tail_mask(norm, 4, 10)
            if axis == 0:
                mask[:, bad] = True
            else:
                mask[bad] = True
    return out


def leading_count(conf):
    # Accept a Counter or a ranked list of confusion records.
    counts = []
    for rec in conf.values() if isinstance(conf, dict) else conf:
        if isinstance(rec, dict):
            rec = rec.get('count', rec.get('n', 0))
        elif isinstance(rec, (list, tuple)) and rec:
            rec = rec[-1]
        if isinstance(rec, (int, float, np.number)):
            counts.append(int(rec))
    return max(counts, default=0)


def main():
    s = open_session('.')
    best = s.measure('C0')
    if abs(best.cer - 0.0726) > 0.0025:
        s.note('C0 outside preregistered tolerance; no repairs.')
        s.freeze()
        return

    halted = False

    def trial(label, text, prediction, op, name, **kw):
        nonlocal best, halted
        if halted or s.budget_left <= 0:
            return False
        h = s.hypothesis(text, prediction, op)
        s.edit(op, [name], hypothesis=h, **kw)
        m = s.measure(label)
        peak0, peak1 = leading_count(best.confusions), leading_count(m.confusions)
        dc = m.cer - best.cer
        regress = m.collapsed > best.collapsed or (peak0 > 0 and peak1 > peak0*1.5 + 10)
        kept = s.keep(f'{label}: CER delta {dc:+.6f}; collapses {best.collapsed}->{m.collapsed}; leading confusion {peak0}->{peak1}; paired rule decides.')
        if kept:
            best = m
            if regress or dc >= 0:
                halted = True
                s.note('Stop at paired keep: auxiliary metrics caution against further adjustments.')
        return kept

    donor = KrakenSubject(PARENT)
    t = s.subject.tensors()
    name = 'L_16.layer.weight_ih_l0'
    p = donor.tensors()[name]
    w = t[name]
    mask = local_mask(w, p)
    if 0 < mask.sum() < w.size * 0.03:
        trial('H1-local', f'{int(mask.sum())} extreme gate-relative residual entries may disrupt final-layer evidence.', 'CER down 0.05-0.8 pp', 'add', name, delta=np.where(mask, p-w, 0).astype(w.dtype))
    else:
        s.note('No isolated L_16 residual meeting local repair criterion.')

    name = 'C_3.co.bias'
    trial('H2-C3', 'Large bias drift relative to parent may mis-threshold early features.', 'CER down 0.01-0.15 pp', 'interpolate', name, other=donor.tensors()[name], alpha=0.5)

    name = 'O_18.lin.bias'
    for label, row, first, next_ok, next_fail, text, pred1, pred2 in [
        ('H3-space', s.subject.codec_map()[' '], 1., .75, .5, 'Missing spaces may have subthreshold classifier peaks.', 'CER down 0.05-0.4 pp', 'CER down 0.01-0.15 pp'),
        ('H4-blank', 0, -.8, -.6, -.35, 'Blank may suppress marginal character peaks across classes.', 'CER down 0.03-0.3 pp', 'CER down 0.01-0.12 pp'),
    ]:
        step = first
        for i, pred in enumerate([pred1, pred2]):
            if halted:
                break
            delta = np.zeros_like(s.subject.tensors()[name])
            delta[row] = step
            ok = trial(f'{label}-{i+1}', f'{text} Add {step:+.2f} to row {row}.', pred, 'add', name, delta=delta)
            step = next_ok if ok else next_fail
    s.freeze()


if __name__ == '__main__':
    main()
