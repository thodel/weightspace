import unicodedata as ud
import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

DONOR = '/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel'
W = 'O_18.lin.weight'

def unit(x):
    return x / max(1e-10, np.linalg.norm(x))

def repair(s, cm):
    donor = KrakenSubject(DONOR)
    pm = donor.codec_map()
    w = s.subject.tensors()[W].astype(np.float64)
    d = donor.tensors()[W].astype(np.float64)
    lower = [a for a in 'anirstouldhgmcbfpkwvz' if a in cm]
    # Map (base, combining marks) to the corresponding precomposed row.
    accents = {}
    for ch in cm:
        n = ud.normalize('NFD', ch)
        if len(n) > 1 and n[0] in 'abcdefghijklmnopqrstuvwxyz' and all(ud.combining(t) for t in n[1:]):
            accents[n[0], n[1:]] = ch

    def candidates(c):
        peers = [a for a in lower if a != c]
        healthy = w[[cm[a] for a in peers]]
        center = healthy.mean(axis=0)
        radius = np.median(np.linalg.norm(healthy-center, axis=1))
        def predict(v):
            return center + radius * unit(v)
        # Contrast an accented row against the SAME mark on other bases,
        # rather than copying a large rare-character row into a frequent slot.
        deltas, weights = [], []
        for (base, mark), ch in accents.items():
            if base != c:
                continue
            controls = [a for a in peers if (a, mark) in accents]
            if not controls:
                continue
            diff = np.mean([unit(w[cm[ch]]) - unit(w[cm[accents[a,mark]]]) +
                            (w[cm[a]]-center)/max(1., np.linalg.norm(w[cm[ch]]))
                            for a in controls], axis=0)
            deltas.append(diff)
            weights.append(len(controls)/(2.+len(controls)))
        out = {}
        if deltas:
            out['marks'] = predict(np.average(deltas, axis=0, weights=weights))
        if c.upper() in cm:
            caps = [a for a in peers if a.upper() in cm]
            if caps:
                out['case'] = predict(unit(w[cm[c.upper()]]) - np.mean([unit(w[cm[a.upper()]]) for a in caps],axis=0))
        if c in pm:
            common = sorted((cm.keys() & pm.keys()) - {'e', c})
            anchor = [a for a in peers if a in pm]
            if anchor and common:
                dc = d[[pm[a] for a in anchor]].mean(axis=0)
                q = d[pm[c]] - dc
                out['direct'] = predict(q)
                x = np.array([unit(d[pm[a]]-dc) for a in common])
                y = np.array([unit(w[cm[a]]-center) for a in common])
                gram = x @ x.T
                coeff = np.linalg.solve(gram + 0.2*np.eye(len(x)), x @ unit(q))
                out['mapped'] = predict(coeff @ y)
        return out, center

    preds, center = candidates('e')
    score = {k: [] for k in preds}
    for proxy in 'aiounrs':
        if proxy not in cm:
            continue
        options, ctr = candidates(proxy)
        truth = unit(w[cm[proxy]]-ctr)
        for k in score.keys() & options.keys():
            score[k].append(float(unit(options[k]-ctr) @ truth))
    # Soft model choice from intact-letter reconstruction, with a small
    # prior count to avoid picking a method on a single lucky match.
    coeff = [np.exp(11*sum(score[k])/(len(score[k])+1.0)) for k in preds]
    vec = np.average(list(preds.values()), axis=0, weights=coeff)
    size = np.median([np.linalg.norm(w[cm[a]]) for a in 'ainrst' if a in cm])
    row = (size * unit(vec)).astype(np.float32)
    h = s.hypothesis('The near-zero lowercase e readout explains the concentration of e deletions.',
                     'Predict a 0.03-0.07 absolute reduction in CER by restoring e evidence.', 'single output row reconstruction')
    s.edit('set_rows', [W], hypothesis=h, index=[cm['e']], values=row[None,:])
    s.keep_unmeasured('Only e has both a missing direction and massive class-specific error; leave its bias and all other rows intact.')

def main():
    s = open_session('.')
    cm = s.subject.codec_map()
    if np.linalg.norm(s.subject.tensors()[W][cm['e']]) < 0.01:
        repair(s, cm)
    s.freeze()

if __name__ == '__main__':
    main()
