import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

DONOR = '/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel'

def unit(x):
    x = x.reshape(len(x), -1).astype(float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)

def match(scores):
    # One-to-one assignment, then improve by all profitable pair exchanges.
    n = len(scores)
    p = np.full(n, -1, dtype=int)
    used = set()
    for k in np.argsort(-scores.ravel()):
        i, j = divmod(int(k), n)
        if p[i] < 0 and j not in used:
            p[i] = j
            used.add(j)
    for _ in range(4*n):
        a = scores[:, p]
        gain = a+a.T-a.diagonal()[:, None]-a.diagonal()[None, :]
        i, j = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[i,j] < 1e-9:
            break
        p[i], p[j] = p[j], p[i]
    return p

def signature(x):
    # Compare spatial templates and spatial covariance even before counterpart
    # channels on the summed-over axis have been put into correspondence.
    x = x.reshape(len(x), x.shape[1], -1).astype(float)
    cov = np.einsum('ncs,nct->nst', x, x)
    def centered(a):
        a = a.reshape(len(a), -1)
        a = unit(a)
        return unit(a-a.mean(0))
    return np.concatenate([centered(x.mean(1)), centered(cov)], axis=1)

def consumer_map(a, b):
    # Returns donor input index for each subject input. Alternately align
    # consuming filters and their input slices; also allow already-aligned outputs.
    si = signature(a.swapaxes(0,1)) @ signature(b.swapaxes(0,1)).T
    so = signature(a) @ signature(b).T
    best = None
    for o in [np.arange(len(a)), match(so)]:
        for _ in range(5):
            ii = match(unit(a.swapaxes(0,1)) @ unit(b[o].swapaxes(0,1)).T + .12*si)
            o = match(unit(a) @ unit(b[:,ii]).T + .12*so)
            score = (unit(a)*unit(b[o][:,ii])).sum() + .12*(si[np.arange(len(ii)),ii].sum()+so[np.arange(len(o)),o].sum())
            if best is None or score > best[0]:
                best = (score, ii)
    return best[1]

def rk(a, b):
    # Uncentered second moment of ReLU projections of a standard Gaussian.
    c = np.clip(unit(a) @ unit(b).T, -1., 1.)
    r = np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)
    return r*(np.sqrt(1-c*c)+(np.pi-np.arccos(c))*c)/(2*np.pi)

def recurrent_sig(t):
    ans = []
    for tail in ['', '_reverse']:
        w = t['L_12.layer.weight_ih_l0'+tail]
        # Choose flattening order by coherence between adjacent height bins.
        choices = [w.reshape(4,200,64,15), w.reshape(4,200,15,64).swapaxes(2,3)]
        x = max(choices, key=lambda a: float(np.mean(a[:,:,:,1:]*a[:,:,:,:-1])))
        for y in [x.mean(1), np.sqrt((x*x).mean(1))]:
            y = y.transpose(1,0,2).reshape(64,-1)
            ans.append(unit(y-y.mean(0)))
    return np.concatenate(ans,axis=1)

def main():
    s = open_session('.')
    base = s.measure('C0')
    S = {k:v.copy() for k,v in s.subject.tensors().items()}
    W = lambda i: 'C_%d.co.weight'%i
    B = lambda i: 'C_%d.co.bias'%i
    if np.any(S[W(3)]) or np.any(S[B(3)]):
        s.note('Expected exactly erased C3 not found; no repair.')
        s.freeze()
        return
    D = KrakenSubject(DONOR).tensors()
    incumbent = base
    terminal = None
    def trial(label, prediction, block, end):
        nonlocal incumbent, terminal
        if not s.budget_left:
            return False
        h = s.hypothesis(label, prediction, 'affine block transplant')
        s.edit('transplant', list(block), hypothesis=h, per_tensor={k:{'donor':v} for k,v in block.items()})
        m = s.measure(label)
        same_blank_profile = (m.collapsed == base.collapsed and str(m.confusions) == str(base.confusions))
        if m.cer < incumbent.cer and (m.collapsed > incumbent.collapsed or same_blank_profile):
            s.revert('Lower CER but worse collapse or unchanged baseline deletion profile')
            return False
        if s.keep('Compare CER with incumbent; collapsed %s -> %s; baseline confusion profile %s'% (incumbent.collapsed,m.collapsed,'unchanged' if str(m.confusions)==str(base.confusions) else 'changed')):
            incumbent, terminal = m, end
            return True
        return False
    def prefix(ids):
        return {k:D[k] for i in ids for k in (W(i), B(i))}
    trial('H1 direct donor C3','CER down 0.03-0.25',prefix([3]),3)
    p = consumer_map(S[W(6)],D[W(6)])
    a = S[W(0)].reshape(32,-1).astype(float)
    b = D[W(0)].reshape(32,-1).astype(float)
    q = match(unit(a) @ unit(b).T)
    dw = D[W(3)][p]
    trial('H2 match both C3 interfaces','CER down 0.01-0.12',{W(3):dw[:,q],B(3):D[B(3)][p]},3)
    g = rk(a,a)
    mix = np.linalg.solve(g + np.eye(len(a))*.025*np.diag(g).mean(),rk(a,b))
    trial('H3 analytic rectified input bridge','CER down 0.01-0.10',{W(3):np.einsum('odhw,sd->oshw',dw,mix), B(3):D[B(3)][p]},3)
    trial('H4 coherent donor through C3','CER down 0.02-0.15',{**prefix([0]),W(3):dw,B(3):D[B(3)][p]},3)
    z = consumer_map(S[W(9)],D[W(9)])
    trial('H5 coherent donor through C6','CER down 0.02-0.15',{**prefix([0,3]),W(6):D[W(6)][z],B(6):D[B(6)][z]},6)
    trial('H6 coherent visual stack','CER down 0.02-0.20',prefix([0,3,6,9]),9)
    z = match(recurrent_sig(S) @ recurrent_sig(D).T)
    trial('H7 CNN matched at recurrent input','CER down 0.01-0.12',{**prefix([0,3,6]),W(9):D[W(9)][z],B(9):D[B(9)][z]},9)
    if terminal is not None:
        for factor in (.65,1.5):
            if not s.budget_left:
                break
            label = 'Interface gain x%s'%factor
            h = s.hypothesis(label,'CER down 0.005-0.05','scale')
            s.edit('scale',[W(terminal),B(terminal)],hypothesis=h,factor=factor)
            m = s.measure(label)
            if m.cer < incumbent.cer and (m.collapsed > incumbent.collapsed or (m.collapsed == base.collapsed and str(m.confusions)==str(base.confusions))):
                s.revert('Collapse/deletion guard')
            elif s.keep('Paired test for gain change relative to best prefix'):
                break
    s.freeze()

if __name__ == '__main__':
    main()
