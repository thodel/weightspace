from wsp.run import open_session
from wsp.pages import read_reference, page_loop
from wsp.metrics import normalise
from pathlib import Path

s = open_session('.')
if s.budget_left != 6 or s._kept_at_cap != 5:
    raise RuntimeError('Expected H1 as last kept state after pass 4')
h = s.hypothesis('If the residual overrun after H1 also depends on end-sensitive axes in the preceding full-attention block, removing the two strong lens outliers in the MLP writers at layers 26 and 27 may shorten excess tails.',
                 'Versus H1: CER -0.035, truncated CER within 0.005 of unchanged, caps -2. Failure gives no support for extending the final-writer edit to this block.',
                 'project_out')
per = {}
for layer, idx in ((26, 2), (27, 0)):
    mod, = [m for m in s.subject.modules() if m.endswith(f'.layers.{layer}.mlp.down_proj')]
    u, _, _ = s.subject.directions(mod)
    per[mod + '.lora_B.weight'] = {'direction': u[:,idx]}
s.edit('project_out', list(per), hypothesis=h, axis=1, per_tensor=per)
s.measure('H4-extend-writers-26-27')
e = s._log[-1]
print(e['label'], e['page'], e['vs_kept'], flush=True)
b = s._kept_lines
ds = [a[0]-p[0] for a,p in zip(s._last_lines,b)]
i = ds.index(min(ds))
print('better/worse/tied', sum(d<0 for d in ds), sum(d>0 for d in ds), ds.count(0), 'drop top beneficiary delta', (sum(ds)-ds[i])/(sum(p[1] for p in b)-b[i][1]), flush=True)
print('H4 kept:', s.keep('Apply registered paired page CER and cap rule.'), flush=True)

# Distinguish resolving prior defects from moving the same count to new pages.
refs = {p['key']:normalise(p['text']) for p in s.practice}
def defects(n):
    rows = read_reference(Path(f'passes/pass-{n:03d}.raw.jsonl'))
    assert set(rows) == set(refs)
    return ({k for k,v in rows.items() if v['at_cap']},
            {k for k,v in rows.items() if page_loop(normalise(v['text']), refs[k])})
for a,b in [(1,2), (2,3), (2,4), (2,5)]:
    print(f'pass {a} to {b}:')
    for label, x, y in zip(('cap','loop'), defects(a), defects(b)):
        print(label, 'resolved', sorted(x-y), 'new', sorted(y-x), flush=True)
