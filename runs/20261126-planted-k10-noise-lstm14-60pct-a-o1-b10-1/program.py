"""Measured repairs for s11. All targets except additive bias steps refer to C0."""
import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'


def rms(x):
    return float(np.sqrt(np.mean(x*x)))


def recover(d, typical):
    """Estimate the hand-specific update, in one IFGO gate, relative to parent."""
    x = d.copy()
    shift = float(np.median(x))
    if abs(shift) > typical:
        x -= shift
    # Only suppress gross outliers; large well-supported fine-tuning values need not be noise.
    mad = float(np.median(np.abs(x-np.median(x)))) * 1.4826
    x[np.abs(x) > 9*max(mad, typical)] = 0
    for axis in (0, 1):
        energy = np.sqrt(np.mean(x*x, axis=axis, keepdims=True))
        x *= np.where(energy > 4*max(typical, float(np.median(energy))), 0., 1.)
    if rms(x) < 1.35*typical:
        return x
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    # A single separated mode explaining a large fraction of the excess energy
    # is more plausibly a coherent offset than isotropic damage.
    gap = np.flatnonzero((sv[:8] > 3*sv[1:9]) & (sv[:8] > typical*np.sqrt(x.size)*0.5))
    if len(gap):
        sv[:int(gap[-1])+1] = 0.
    total_var = float(sv@sv)/x.size
    if total_var <= (1.35*typical)**2:
        return (u*sv)@vt
    # Good task updates can have singular modes above the rectangular iid bulk.
    # Keep such modes more strongly than the diffuse update.
    var = total_var-typical**2
    m,n = x.shape
    edge = np.sqrt(var)*(np.sqrt(m)+np.sqrt(n))
    strong = sv > edge
    coeff = np.zeros_like(sv)
    coeff[strong] = np.sqrt(np.maximum((sv[strong]**2 - (m+n)*var)**2 - 4*m*n*var**2, 0))/sv[strong]
    diffuse = typical**2/total_var
    return (u*(diffuse*sv + (1-diffuse)*coeff))@vt


def pairs(m):
    # Accept either counted pairs, triples, or labelled confusion records.
    data = m.confusions
    items = data.items() if isinstance(data, dict) else data
    out = {}
    empty = (None, '', '∅', '<del>', '<ins>', '<eps>')
    for e in items or ():
        if isinstance(e, dict):
            a = e.get('ref', e.get('reference')); b = e.get('hyp', e.get('hypothesis')); n = e.get('count', e.get('n', 0))
        elif isinstance(e, (tuple,list)) and len(e)==3:
            a,b,n = e
        elif isinstance(e, (tuple,list)) and len(e)==2:
            key,n = e
            if isinstance(key, (tuple,list)) and len(key)==2:
                a,b = key
            elif isinstance(key,str) and '→' in key:
                a,b = key.split('→',1)
            else:
                continue
        else:
            continue
        if isinstance(n,(int,float,np.number)) and (a not in empty or b not in empty):
            out[('' if a in empty else a, '' if b in empty else b)] = int(n)
    return out


def main():
    s = open_session('.')
    current = s.measure('C0')
    p = KrakenSubject(PARENT).tensors()
    t = s.subject.tensors()
    names = sorted(k for k in t if k.startswith('L_14.layer.weight_'))
    orig = {k:t[k].copy() for k in names}
    fix = {}
    for k in names:
        gates = np.split(orig[k]-p[k],4)
        refs = [np.split(t[j]-p[j],4) for j in (k.replace('L_14','L_12'),k.replace('L_14','L_16'))]
        clean = [recover(g, max(1e-6,np.mean([rms(r[i]) for r in refs]))) for i,g in enumerate(gates)]
        fix[k] = p[k]+np.concatenate(clean)

    def test(label, text, prediction, op, args):
        nonlocal current
        if not s.budget_left:
            return False
        h = s.hypothesis(text,prediction,op)
        s.edit(op,list(args),hypothesis=h,per_tensor=args)
        m = s.measure(label)
        prev, now = pairs(current), pairs(m)
        spike = max(now.values(),default=0) > 1.5*max(prev.values(),default=0)+20
        if m.collapsed > current.collapsed or spike:
            s.revert('Veto: collapse count rose or largest confusion spiked.')
            return False
        why = 'Lower CER without collapse or confusion spike' if m.cer < current.cer else 'No measured CER gain'
        if s.keep(why+'; require paired interval below zero.'):
            current = m
            return True
        return False

    def matrices(label,text,pred,target):
        return test(label,text,pred,'add',{k:{'delta':np.asarray(target[k]-s.subject.tensors()[k],np.float32)} for k in names})

    adaptive = matrices('H1 gate denoising','L14 excess residual obscures the learned hand-specific map.','CER down 1–5 pp',fix)
    def blend(alpha):
        target = {k:(1-alpha)*orig[k]+alpha*p[k] for k in names}
        return matrices('H2 parent %.2f'%alpha,'A uniform donor blend may better suppress excess L14 noise.','CER down 0.1–1 pp after repair or 1–4 pp from C0',target)
    heavy = blend(0.90)
    alt = blend(1.0 if heavy or not adaptive else 0.55)
    if not adaptive or heavy or alt:
        other = s.subject.tensors() if adaptive else orig
        matrices('H3 compromise','An intermediate dose retains useful detail lost with a stronger correction.','CER down 0.1–1 pp',{k:0.5*(other[k]+fix[k]) for k in names})
    k = 'C_3.co.bias'
    test('H4 C3 thresholds','C3 bias drift could disrupt early thresholds.','CER down 0.02–0.4 pp','transplant',{k:{'donor':p[k]}})
    k = 'O_18.lin.bias'
    def bias(label,row,step):
        d = np.zeros_like(s.subject.tensors()[k]); d[row] = step
        return test(label,'Deletion-heavy confusions reflect a logit prior error: '+label,'CER down 0.05–0.5 pp','add',{k:{'delta':d}})
    def space_heavy():
        c = pairs(current)
        return c.get((' ',''),0) > 1.5*c.get(('',' '),0)+8
    cm = s.subject.codec_map()
    if ' ' in cm and space_heavy():
        if bias('H5 space +0.8',cm[' '],0.8) and space_heavy():
            bias('H5 space +0.7',cm[' '],0.7)
    top = sorted(pairs(current).items(),key=lambda it:-it[1])[:10]
    if sum(n for (a,b),n in top if b=='') > sum(n for (a,b),n in top if b!=''):
        bias('H6 blank -0.8',0,-0.8)
    s.freeze()


if __name__ == '__main__':
    main()
