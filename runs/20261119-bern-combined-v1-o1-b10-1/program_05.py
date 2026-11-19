from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

s = open_session('.')
assert s.budget_left == 2
h = s.hypothesis('Test the other explanation for the pass-7 loss: early visual features may need *more* Schlussredaktion adaptation, not more parent weight. Move the convolution stack and first recurrent layer 50% toward the clean sibling; keep both upper recurrent layers and head exactly at pass 2 so its recovered Hertwig context is not removed. The parent saw all 103 correct Hertwig practice lines, the sibling fine-tuned on all 197 clean Schlussredaktion practice lines. Effects need not transfer.', 'Vs pass 2 expect Schlussredaktion down 0.6–1.3 CER points with Hertwig flat to +0.7, pooled down about 0.2–0.8 points. Failure means the sibling visual stack at this dose does not yield a reliable further gain with preserved upper layers.', 'early visual sibling interpolation')
sib = KrakenSubject('/home/tobias/Repo/weightspace/models/kraken/kraken-bern-schlussredaktion-v1.mlmodel')
names = [k for k in s.subject.tensors() if k.startswith(('C_', 'L_12.'))]
b = sib.tensors()
s.edit('interpolate', names, hypothesis=h, alpha=0.5, per_tensor={k:{'other':b[k]} for k in names})
m = s.measure('early convs and L12 halfway toward sibling, upper layers unchanged')
print('CER', m.cer, 'collapse', m.collapsed, 'length ratio', m.length_ratio, 'vs kept', s._log[-1]['vs_kept'])
for doc in sorted({it['key'].split('/')[0] for it in s.practice}):
    e,n = np.sum([line for line,it in zip(s._last_lines,s.practice) if it['key'].split('/')[0] == doc],axis=0)
    print(doc,e,n,e/n)
if m.collapsed > 23:
    s.revert('Collapsed lines increased vs 23 in pass 2')
    print('reverted on collapse guard')
else:
    print('kept', s.keep('Apply paired CER rule after collapse check'))
s.note('Stopping after the planned early-sibling contrast. Leave the last pass unused rather than select an additional dose from the same practice lines.')
s.freeze()
