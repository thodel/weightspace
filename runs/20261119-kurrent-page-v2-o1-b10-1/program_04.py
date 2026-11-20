from wsp.run import open_session

s = open_session('.')  # restores H1, not the last measured candidate
if s.budget_left != 5 or s._kept_at_cap != 5:
    raise RuntimeError('Expected five passes completed, with H1 last kept.')
s.note('Early stop after H4: none of H2-H4 resolved a remaining H1 cap. Freeze the layer-31 edit; no further practice search.')
s.freeze()
