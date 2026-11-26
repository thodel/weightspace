import unicodedata
import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

DONOR = '/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel'
W, B = 'O_18.lin.weight', 'O_18.lin.bias'


def base(c):
    return ''.join(k for k in unicodedata.normalize('NFD', c) if not unicodedata.combining(k))


def unit(x):
    return x / max(np.linalg.norm(x), 1e-8)


def estimates(w, cm, donor):
    # Learn coordinate-wise weight analogies by hiding intact common rows in turn.
    # No use of line images, activations or transcription targets.
    pm = donor.codec_map()
    dw = donor.tensors()[W].astype(float)
    letters = [c for c in 'aeinrstduhglombwkfzcpvjyxq' if c in cm]
    norm = np.median([np.linalg.norm(w[cm[c]]) for c in 'ainsrt' if c in cm])
    def make(c):
        rem = [k for k in letters if k not in ('e', c)]
        r = w[[cm[k] for k in rem]]
        other = np.delete(w, list({cm[c], cm['e'], 0}), axis=0)
        pairs = [k for k in rem if k in pm]
        if c in pm and pairs:
            # Express the donor character as a regularized combination of donor
            # anchors; carry coefficients, not hidden-unit indices, to the subject.
            x = np.stack([unit(dw[pm[k]]) for k in pairs]); y = unit(dw[pm[c]])
            a = np.linalg.solve(x @ x.T + 0.22*np.eye(len(x)), x @ y)
            transfer = a @ w[[cm[k] for k in pairs]]
        else:
            transfer = r.mean(0)
        def variants(k):
            v = [unit(w[i])*norm for ch,i in cm.items() if ch != k and base(ch) == k]
            return np.mean(v, axis=0) if v else None
        # Estimate systematic displacement of accent rows from plain vowels.
        vc = variants(c)
        shifts = [variants(k)-w[cm[k]] for k in rem if k in 'aiouy' and variants(k) is not None]
        accent = (vc - np.mean(shifts, axis=0) if vc is not None and shifts else
                  vc if vc is not None else transfer)
        group = 'aeiou' if c in 'aeiou' else ('mnhr' if c in 'mnhr' else 'tsflkbd')
        near = [cm[k] for k in rem if k in group]
        peers = w[near].mean(0) if near else r.mean(0)
        feats = [transfer, accent, peers, w[0], r.mean(0), other.mean(0),
                 r.max(0), r.min(0), np.median(r,axis=0), other.max(0), other.min(0),
                 np.quantile(other,.1,axis=0), np.quantile(other,.9,axis=0)]
        return np.stack(feats,axis=1), accent, peers
    xs, ys = [], []
    for c in letters:
        if c == 'e': continue
        # Common vowels are the most relevant proxy tasks.
        n = 3 if c in 'aiou' else 1
        xs.extend([make(c)[0]]*n); ys.extend([w[cm[c]]]*n)
    x = np.concatenate(xs); y = np.concatenate(ys)
    mu, sd = x.mean(0), np.maximum(x.std(0),1e-6)
    z = (x-mu)/sd
    beta = np.linalg.solve(z.T@z + .015*len(z)*np.eye(z.shape[1]), z.T@(y-y.mean()))
    f, acc, peers = make('e')
    pred = ((f-mu)/sd)@beta + y.mean()
    # Preserve some shrinkage: unsupported components should not have full gain.
    pred *= min(2., .9*norm/max(np.linalg.norm(pred),1e-8))
    analog = .78*unit(acc)*norm + .22*peers
    boundary = .48*w[0] + .52*sum(p*w[cm[k]] for k,p in zip('aiorcu',[.35,.24,.14,.10,.09,.08]) if k in cm)
    return pred, analog, boundary, norm


def balance(m):
    # Count the displayed confusion events involving e (truncated lists are a
    # directional diagnostic only). Support pair-key counts or serialized rows.
    out = [0,0]
    obj = m.confusions
    rows = obj.items() if isinstance(obj,dict) else obj
    for r in rows or []:
        if isinstance(r,dict):
            a,b,n = r.get('ref'),r.get('hyp'),r.get('count',r.get('n',0))
        elif len(r)==3: a,b,n = r
        elif len(r)==2 and isinstance(r[0],(list,tuple)) and len(r[0])==2: (a,b),n = r
        else: continue
        if a=='e' and b!='e': out[0]+=int(n)
        if b=='e' and a!='e': out[1]+=int(n)
    return out


def main():
    s = open_session('.')
    kept = s.measure('C0')
    cm = s.subject.codec_map(); i = cm['e']
    w = s.subject.tensors()[W].astype(float); b = float(s.subject.tensors()[B][i])
    pred, acc, bound, norm = estimates(w,cm,KrakenSubject(DONOR))
    def test(label, row, bias, effect):
        nonlocal kept
        names = [B] if row is None else [W,B]
        args = {B:{'values':np.array([bias],dtype=np.float32)}}
        if row is not None: args[W]={'values':np.asarray(row,dtype=np.float32)[None,:]}
        h = s.hypothesis(label, effect, 'set_rows: e only')
        s.edit('set_rows', names, hypothesis=h, index=[i], per_tensor=args)
        m=s.measure(label); miss, extra=balance(m)
        why=f'CER {kept.cer:.5f}->{m.cer:.5f}; collapsed {kept.collapsed}->{m.collapsed}; reported e losses/gains {miss}/{extra}'
        if m.collapsed>kept.collapsed+1 or (m.cer<kept.cer and extra-miss>500 and kept.cer-m.cer<.005):
            s.revert('guard: '+why); return False
        if s.keep(why): kept=m; return True
        return False
    if 1e-12 < np.linalg.norm(w[i]) < .1:
        test('Restore attenuated e direction',unit(w[i])*norm,b,'CER down 2-8 pp')
    for label,row,offset in [('Reconstruct e using intact-row analogies',pred,2.5),('Subject deaccented vowel analogy',acc,2.5),('Blank and competitor boundary',bound,3.0)]:
        test(label,row,b+offset,'CER down 1-8 pp vs C0 or 0.2-2 pp further')
    center=float(s.subject.tensors()[B][i]); tried={round(center,3)}
    miss,extra=balance(kept); sign=1 if miss>=extra else -1
    for shift in [2.5*sign,5*sign,-2.5*sign,7.5*sign]:
        if s.budget_left<=1: break
        v=center+shift; tried.add(round(v,3))
        test('Calibrate e threshold %+.2f'%shift,None,v,'CER down 0.3-3 pp if residual errors are threshold-driven')
    for step in [1.25,.65]:
        if not s.budget_left: break
        miss,extra=balance(kept); direction=1 if miss>=extra else -1
        current=float(s.subject.tensors()[B][i]); v=current+direction*step
        if round(v,3) in tried: v=current-direction*step
        tried.add(round(v,3)); test('Refine e threshold at %.3f'%v,None,v,'CER down about 0.3 pp')
    s.freeze()


if __name__=='__main__': main()
