from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

s = open_session('.')
assert s.budget_left == 8
collapse_limit = 23

def decide(label):
    global collapse_limit
    m = s.measure(label)
    print(label, 'CER', m.cer, 'collapsed', m.collapsed, 'length ratio', m.length_ratio, 'vs kept', s._log[-1]['vs_kept'])
    for doc in sorted({x['key'].split('/')[0] for x in s.practice}):
        lines = [line for x, line in zip(s.practice, s._last_lines) if x['key'].split('/')[0] == doc]
        e, n = np.sum(lines,axis=0)
        print(doc, e, n, e/n)
    if m.collapsed > collapse_limit:
        s.revert('Collapsed count increased vs kept')
        ok = False
    else:
        ok = s.keep('Apply paired bootstrap rule after collapse check')
    print('kept', ok, flush=True)
    if ok:
        collapse_limit = m.collapsed

h = s.hypothesis('Keep the blended stroke features but move the final recurrent context and classifier 40% toward the sibling. The parent saw the 103 clean Hertwig practice lines; the sibling adapted from it on clean Schlussredaktion including the 197 practice lines. The latter lost 68 edits in the blend and case substitutions rose. Training-side test only.', 'vs pass 2: Schlussredaktion down 0.7–1.6 points, Hertwig flat to +1 point; pooled down 0.3–1 point. Failure means this upper-layer tilt does not recover enough of the second hand without sacrificing the first.', 'upper-block sibling interpolation')
sib = KrakenSubject('/home/tobias/Repo/weightspace/models/kraken/kraken-bern-schlussredaktion-v1.mlmodel')
t, dt = s.subject.tensors(), sib.tensors()
cm, dm = s.subject.codec_map(), sib.codec_map()
args = {}
for k, w in t.items():
    if k.startswith('L_16.'):
        args[k] = {'delta': 0.4 * (dt[k]-w)}
    elif k.startswith('O_18.'):
        delta = np.zeros_like(w)
        delta[0] = 0.4 * (dt[k][0]-w[0])
        for c in cm.keys() & dm.keys():
            delta[cm[c]] = 0.4 * (dt[k][dm[c]]-w[cm[c]])
        args[k] = {'delta': delta}
s.edit('add', list(args), hypothesis=h, per_tensor=args)
decide('L16 and head 40% toward clean sibling')

h = s.hypothesis('Short readings and excess deletions may reflect character peaks losing to blank. Lower blank bias by 0.8 logit on the last kept model; no further donor change. A failure suggests missing characters are not recoverable by this uniform bias shift.', 'Pooled CER down 0.2–0.8 percentage points vs last kept, with length ratio up. Both hands are short; expect fewer deletions on each.', 'blank offset')
k = 'O_18.lin.bias'
delta = np.zeros_like(s.subject.tensors()[k]); delta[0] = -0.8
s.edit('add', [k], hypothesis=h, delta=delta)
decide('blank bias -0.8')
