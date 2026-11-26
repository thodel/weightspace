import numpy as np
from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'


def main():
    s = open_session('.')
    best = s.measure('C0')
    pt = KrakenSubject(PARENT).tensors()
    t = s.subject.tensors()

    def severe():
        return best.collapsed >= 12 or best.cer >= .25

    def trial(label, text, prediction, op, name, **kwargs):
        nonlocal best
        if s.budget_left <= 0:
            return False
        h = s.hypothesis(text, prediction, op + ': ' + name)
        s.edit(op, [name], hypothesis=h, **kwargs)
        m = s.measure(label)
        # Confusion entries may be truncated to the most frequent pairs. Use
        # these as diagnostics, not as if their sum were total error count.
        same = str(m.confusions) == str(best.confusions)
        msg = ('CER %.4f -> %.4f; collapsed %d -> %d; leading confusions %s: %s'
               % (best.cer, m.cer, best.collapsed, m.collapsed,
                  'unchanged' if same else 'changed', str(m.confusions)[:280]))
        if m.cer < best.cer and m.collapsed > best.collapsed + 3:
            s.revert('More lines lost despite aggregate improvement: ' + msg)
            return False
        kept = s.keep(msg)
        if kept:
            best = m
        return kept

    bias = 'C_3.co.bias'
    d = t[bias].copy() - pt[bias]
    # Prefer a repeated nonzero delta over the median if some channels
    # escaped the perturbation. Fine tuning need not give exact ties.
    u, counts = np.unique(np.round(d, 6), return_counts=True)
    j = int(np.argmax(counts))
    offset = float(np.median(d))
    mask = np.ones_like(d, dtype=bool)
    if counts[j] >= 4 and abs(u[j]) > 1e-6:
        offset = float(u[j])
        mask = np.abs(d) > 1e-7
    if abs(offset) > 1e-6:
        trial('C3 common threshold', 'C3 thresholds moved together relative to the donor.',
              'CER down .02-.35, collapse down', 'add', bias,
              delta=(-offset * mask).astype(d.dtype))
    if severe() and np.max(np.abs(t[bias] - pt[bias])) > 1e-6:
        trial('C3 thresholds', 'Mis-set individual C3 thresholds suppress visual evidence.',
              'CER down .005-.35', 'transplant', bias, donor=pt[bias])

    candidates = []
    if severe():
        for n, x in t.items():
            if n.startswith('O_') or x.ndim < 2 or x.shape != pt[n].shape:
                continue
            for ax in (0, 1):
                a = np.moveaxis(x, ax, 0).reshape(x.shape[ax], -1)
                b = np.moveaxis(pt[n], ax, 0).reshape(a.shape)
                if min(a.shape) < 3:
                    continue
                pn = np.linalg.norm(b, axis=1)
                en = np.linalg.norm(a-b, axis=1)
                # Denominator floor avoids treating near-dead donor filters
                # as major damage. Require large absolute AND relative drift.
                rel = en / np.maximum(pn, .25*np.median(pn)+1e-7)
                strength = np.minimum(rel / max(.08, np.median(rel)),
                                      en / max(1e-6, np.median(en)))
                bad = (strength >= 3.3) & (rel > .70)
                if not np.any(bad) or bad.mean() > .18:
                    continue
                target = np.moveaxis(x.copy(), ax, 0)
                target[bad] = np.moveaxis(pt[n], ax, 0)[bad]
                candidates.append((float(strength[bad].max()), n, ax,
                                   np.moveaxis(target, 0, ax)-x))
    used = set()
    for score, n, ax, change in sorted(candidates, key=lambda c: -c[0]):
        if not severe() or len(used) >= 2:
            break
        if n in used:
            continue
        used.add(n)
        trial('slice repair '+n, 'Isolated axis-%d slices of %s have unusually large parent residuals (%.1fx typical).' % (ax,n,score),
              'CER down .01-.40', 'add', n, delta=change)

    if severe():
        options = []
        for k in (0, 3, 6, 9):
            n = 'C_%d.co.weight' % k
            r = t[n] - pt[n]
            dc = r.mean(axis=(2,3), keepdims=True)
            z = float(np.sqrt(r.shape[2]*r.shape[3])*np.linalg.norm(dc) /
                      max(1e-9, np.linalg.norm(r)/np.sqrt(r.shape[2]*r.shape[3])))
            options.append((z, n, dc))
        z, n, dc = max(options, key=lambda p:p[0])
        trial('kernel DC', 'Spatially constant drift most enriched in %s (%.2fx isotropic RMS).' % (n,z),
              'CER down .01-.30', 'add', n, delta=np.broadcast_to(-dc,t[n].shape).copy())
    for k in (3, 0):
        if severe():
            n = 'C_%d.co.weight' % k
            trial('visual blend '+n, 'Diffuse early filter changes suppress evidence before sequence recognition.',
                  'CER down .01-.25', 'interpolate', n, other=pt[n], alpha=.5)

    if severe() and best.collapsed >= 10:
        n = 'O_18.lin.bias'
        def adjust(amount):
            v = np.zeros_like(t[n]); v[0] = amount
            return trial('blank shift %g'%amount, 'Blank logit wins on weak strokes, causing shared deletions.',
                         'CER down .005-.12; fewer collapsed lines', 'add', n, delta=v)
        ok = adjust(-2.)
        if severe() and best.collapsed >= 10:
            adjust(-2. if ok else -.75)
    s.freeze()


if __name__ == '__main__':
    main()
