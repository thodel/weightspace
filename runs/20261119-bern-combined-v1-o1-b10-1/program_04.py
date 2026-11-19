from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np
s=open_session('.')
assert s.budget_left == 4
limit=23

def decide(label):
    global limit
    m=s.measure(label)
    print(label,m.cer,m.collapsed,m.length_ratio,s._log[-1]['vs_kept'])
    for doc in sorted({x['key'].split('/')[0] for x in s.practice}):
        e,n=np.sum([v for v,x in zip(s._last_lines,s.practice) if x['key'].split('/')[0]==doc],axis=0)
        print(doc,e,n,e/n)
    if m.collapsed>limit:
        s.revert('Collapse rose vs kept'); ok=False
    else: ok=s.keep('Apply paired interval rule')
    if ok: limit=m.collapsed
    print('kept',ok,flush=True)
    return ok

h=s.hypothesis('Test whether the parent contribution is needed principally for visual features rather than deep context. Push convs and L12 halfway toward the clean Hertwig parent while pushing L14, L16 and head halfway toward the clean Schlussredaktion sibling. The parent saw all 103 correctly paired Hertwig practice lines and the sibling fine-tuned on all 197 Schlussredaktion practice lines. A gain could reflect training exposure.', 'From pass 2: pooled -0.3 to -1.2 points, expecting Hertwig no worse than +0.5 and Schlussredaktion down 0.5–1.5 points; failure rejects this division of labor at these doses.', 'depth-dependent donor interpolation')
root='/home/tobias/Repo/weightspace/models/kraken/kraken-bern-'
parent=KrakenSubject(root+'hertwig-v2.mlmodel'); sib=KrakenSubject(root+'schlussredaktion-v1.mlmodel'); p=parent.tensors(); b=sib.tensors(); cm=s.subject.codec_map(); dm=sib.codec_map(); t=s.subject.tensors()
a={}
for k,w in t.items():
    if k.startswith('O_18.'):
        d=np.zeros_like(w); d[0]=0.5*(b[k][0]-w[0]);
        for c in cm.keys() & dm.keys(): d[cm[c]]=0.5*(b[k][dm[c]]-w[cm[c]])
    else:
        src=p if k.startswith(('C_', 'L_12.')) else b
        d=0.5*(src[k]-w)
    a[k]={'delta':d}
s.edit('add', list(a), hypothesis=h, per_tensor=a)
ok=decide('early visual parent / late context sibling')

# Distinct test: restore the actual subject, *not* the sibling, at the final two layers.
h=s.hypothesis('The unedited subject was fine-tuned on clean Schlussredaktion and predominantly incorrect Hertwig pairings. Its output conventions may be useful on the majority hand even if its lower representation suffers. Restore its final recurrent layer and its character-aligned head, retaining the blended lower stack. These starting blends used a parent trained on all 103 clean Hertwig practice lines and a sibling fine-tuned on all 197 clean Schlussredaktion lines.', 'Vs last kept: recover 0.6–1.4 CER points on Schlussredaktion with Hertwig loss under 0.6 points; pooled reduction 0.2–0.9 points. Failure suggests subject upper layers cannot recover the conventions at acceptable cost.', 'subject upper-layer restoration')
orig=KrakenSubject(root+'combined-v1.mlmodel'); o=orig.tensors(); om=orig.codec_map(); a={}
for k,w in s.subject.tensors().items():
    if k.startswith('L_16.'): a[k]={'donor':o[k]}
    elif k.startswith('O_18.'):
        v=w.copy(); v[0]=o[k][0]
        for c in cm.keys() & om.keys(): v[cm[c]]=o[k][om[c]]
        a[k]={'donor':v}
s.edit('transplant',list(a),hypothesis=h,per_tensor=a)
decide('restore subject L16 and head')
