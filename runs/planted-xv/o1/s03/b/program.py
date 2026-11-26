import numpy as np
from wsp.run import open_session

s = open_session('.')
name = 'O_18.lin.bias'
delta = np.zeros_like(s.subject.tensors()[name])
delta[0] = -2.0
h = s.hypothesis('Excess blank intercept suppresses weak nonblank peaks.',
                 'Fewer deletions; practice CER lower by 1-3 percentage points.',
                 'add to blank bias only')
s.edit('add', [name], hypothesis=h, delta=delta)
s.keep_unmeasured('The deletion imbalance and elevated blank intercept support a localized correction.')
s.freeze()
