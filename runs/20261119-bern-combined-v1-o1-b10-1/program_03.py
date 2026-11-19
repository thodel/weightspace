from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

s = open_session('.')
assert s.budget_left == 6
limit = 23

def test(label):
    global limit
    m = s.measure(label)
    print(label, m.cer, m.length_ratio, m.collapsed, s._log[-1]['vs_kept'])
    for doc in sorted({it['key'].split('/')[0] for it in s.practice}):
        e, n = np.sum([v for it,v in zip(s.practice,s._last_lines) if it['key'].split('/')[0]==doc], axis=0)
        print(doc, e, n, e/n)
    if m.collapsed > limit:
        s.revert('Collapse rose relative to last kept'); kept=False
    else:
        kept = s.keep('Apply paired line-level rule')
    if kept: limit=m.collapsed
    print('kept',kept,flush=True)

h=s.hypothesis('Replace the remaining subject contribution with clean donor weights, rather than just tilting the final recurrent block. The parent trained on the 103 Hertwig practice lines with correct images; the sibling fine-tuned on the 197 clean Schlussredaktion lines. A 40% parent/60% sibling mix may preserve both adaptations while excluding mismatched-pair training. No test-side claim.', 'Against pass 2 predict Hertwig within +/-1 point, Schlussredaktion down 0.6–1.5 points, pooled down 0.3–1 point. Failure would mean that dropping subject weights at this balance is not a demonstrated improvement.', 'aligned whole-network donor mixture')
t=s.subject.tensors(); cm=s.subject.codec_map(); delta={k:np.zeros_like(w) for k,w in t.items()}
for tag,coef in [('hertwig-v2',0.4),('schlussredaktion-v1',0.6)]:
    donor=KrakenSubject('/home/tobias/Repo/weightspace/models/kraken/kraken-bern-'+tag+'.mlmodel'); dt=donor.tensors(); dm=donor.codec_map()
    for k,w in t.items():
        if k.startswith('O_18.'):
            delta[k][0]+=coef*(dt[k][0]-w[0])
            for c in cm.keys() & dm.keys():
                delta[k][cm[c]]+=coef*(dt[k][dm[c]]-w[cm[c]])
        else: delta[k]+=coef*(dt[k]-w)
s.edit('add', list(delta), hypothesis=h, per_tensor={k:{'delta':v} for k,v in delta.items()}); test('clean blend 40H/60S; missing donor rows retain current share')

h=s.hypothesis('Lower the blank threshold more for letters than for separators: the uniform -0.8 recovered letters but slash insertions rose beyond deletions. Also raise capital D versus lowercase d, a frequent directional case error. Apply to the last kept head only.', 'Against last kept predict 0.3–0.9 points lower pooled practice CER, fewer deleted letters on both hands, fewer D-to-d substitutions; a failure would reject this fixed set of offsets as a joint correction.', 'class-specific CTC bias calibration')
k='O_18.lin.bias'; d=np.zeros_like(s.subject.tensors()[k]); d[0]=-1.0; d[cm[' ']]=-0.5; d[cm['/']]=-0.95; d[cm['D']]=1.4
s.edit('add',[k],hypothesis=h,delta=d);test('blank -1, space -0.5, slash -0.95, D +1.4')
