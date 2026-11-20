from wsp.run import open_session

s = open_session('.')
if s.budget_left != 10:
    raise RuntimeError('program_01 requires a fresh budget')
s.measure('C0')
page = s._log[-1]['page']
expected = dict(pages=64, chars=108243, cer=0.45151187605665033,
                truncated_cer=0.2563860942508984, length_ratio=1.2263702964625889,
                at_cap=9, loops=12)
print('C0', page, flush=True)
if any(page[k] != v for k, v in expected.items()):
    s.note('C0 did not reproduce archived practice scores. Stopped before any edit.')
    raise RuntimeError('C0 mismatch; see page scores above')

h = s.hypothesis('Leading output axes of the final attention and MLP updates may jointly sustain transcription past the last line.',
                 'CER -0.055, truncated CER unchanged within 0.005, cap count -3 vs C0.',
                 'project_out')
per = {}
for suffix in ('.layers.31.self_attn.o_proj', '.layers.31.mlp.down_proj'):
    mod, = [m for m in s.subject.modules() if m.endswith(suffix)]
    u, _, _ = s.subject.directions(mod)
    per[mod + '.lora_B.weight'] = {'direction': u[:, 0]}
s.edit('project_out', list(per), hypothesis=h, per_tensor=per, axis=1)
s.measure('H1-final-writers')
print('H1', s._log[-1]['page'], s._log[-1]['vs_kept'], flush=True)
print('H1 kept:', s.keep('Apply paired page CER and cap rule for H1.'), flush=True)
