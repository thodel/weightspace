from wsp.run import open_session

s = open_session('.')
if s.budget_left != 8 or s._kept_at_cap != 5:
    raise RuntimeError('Expected two passes completed and H1 as last kept state')

def test(label, kind, layers, factor, text, pred):
    names = []
    for layer in layers:
        suffix = f'.layers.{layer}.{kind}'
        mod, = [m for m in s.subject.modules() if m.endswith(suffix)]
        names.append(mod + '.lora_B.weight')
    h = s.hypothesis(text, pred, 'scale')
    s.edit('scale', names, factor=factor, hypothesis=h)
    s.measure(label)
    print(label, s._log[-1]['page'], s._log[-1]['vs_kept'], flush=True)
    pairs = list(zip(s._last_lines, s._kept_lines))
    ds = [a[0]-b[0] for a,b in pairs]
    i = ds.index(min(ds))
    print('pages better/worse/tied', sum(d<0 for d in ds), sum(d>0 for d in ds), ds.count(0),
          'delta CER excluding largest beneficiary', (sum(ds)-ds[i])/sum(a[1] for j,(a,b) in enumerate(pairs) if i!=j), flush=True)
    print(label, 'kept:', s.keep(f'Apply registered paired page CER and cap rule for {label}.'), flush=True)

test('H2-late-queries', 'self_attn.q_proj', (19, 23, 27, 31), 0.65,
     'Reducing late query adaptation may weaken attention to already-generated text that sustains repetition past the last line.',
     'Relative to H1, CER -0.045, truncated CER +0.003, caps -2.')
test('H3-late-gates', 'mlp.gate_proj', range(24, 32), 0.75,
     'Large late gate updates may maintain transcription features after the last visible line; attenuate those gates only.',
     'Relative to the last kept state, CER -0.040, truncated CER +0.005, caps -2.')
