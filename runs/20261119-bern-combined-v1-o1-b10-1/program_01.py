from pathlib import Path
import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

s = open_session('.')
assert s.budget_left == 10, 'Round 0 program requires a fresh session'
c0 = s.measure('C0')
print('C0 CER, collapsed:', c0.cer, c0.collapsed, flush=True)
if abs(c0.cer - 1577/10829) > 1e-7 or c0.collapsed != 29:
    s.note('C0 mismatch; stopping before any edit')
    raise SystemExit('C0 mismatch; expected 1577/10829 and 29 collapsed')

h = s.hypothesis('Average the two fine-tunes while shrinking toward their common parent across the entire network. Parent trained on correct Hertwig including 103 practice lines; sibling fine-tuned on clean Schlussredaktion including 197 practice lines. Practice gain can reflect donor training exposure.', '35% subject, 30% parent, 35% sibling: pooled CER down 1–2 points; Hertwig down 3–6, Schlussredaktion within half a point of C0.', 'aligned whole-network interpolation')
root = Path('/home/tobias/Repo/weightspace/models/kraken')
t = s.subject.tensors()
cm = s.subject.codec_map()
deltas = {k: np.zeros_like(w) for k, w in t.items()}
for tag, fraction in [('hertwig-v2', 0.30), ('schlussredaktion-v1', 0.35)]:
    donor = KrakenSubject(str(root / ('kraken-bern-' + tag + '.mlmodel')))
    dt, dm = donor.tensors(), donor.codec_map()
    pairs = [(0, 0)] + [(cm[c], dm[c]) for c in cm.keys() & dm.keys() if cm[c] != 0 and dm[c] != 0]
    dest, src = map(list, zip(*pairs))
    for k, w in t.items():
        if k.startswith('O_18.'):
            deltas[k][dest] += fraction * (dt[k][src] - w[dest])
        else:
            deltas[k] += fraction * (dt[k] - w)
s.edit('add', list(t), hypothesis=h, per_tensor={k: {'delta': d} for k, d in deltas.items()})
m = s.measure('35 subject + 30 parent + 35 sibling')
print('CER, length ratio, collapsed, vs kept:', m.cer, m.length_ratio, m.collapsed, s._log[-1]['vs_kept'], flush=True)
for doc in sorted({x['key'].split('/')[0] for x in s.practice}):
    lines = [line for x, line in zip(s.practice, s._last_lines) if x['key'].split('/')[0] == doc]
    e, n = np.sum(lines, axis=0)
    print(doc, e, n, e/n)
if m.collapsed > c0.collapsed:
    s.revert('More collapsed lines than C0')
    print('reverted: collapse guard')
else:
    print('kept:', s.keep('Apply paired line-bootstrap rule after collapse guard'))
