from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = '/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel'
WEIGHT = 'O_18.lin.weight'

s = open_session('.')
parent = KrakenSubject(PARENT)
source = parent.codec_map()['e']
target = s.subject.codec_map()['e']
donor_row = parent.tensors()[WEIGHT][[source]].copy()
h = s.hypothesis(
    'The zero lowercase e output weight row makes its score independent of the handwriting.',
    'Restoring the parent row should lower CER by 8–11 percentage points, mainly on e.',
    'character-aligned single weight row transplant',
)
s.edit('set_rows', [WEIGHT], hypothesis=h, index=[target], values=donor_row)
s.keep_unmeasured('The erased row matches the dominant reference-side errors; the child bias and all other rows are intact and left unchanged.')
s.freeze()
