from wsp.run import open_session

s = open_session('.')
name = 'O_18.lin.weight'
row = int(s.subject.codec_map()['n'])
h = s.hypothesis(
    'The n weight row is sign-reversed: cosine -0.991 to parent and 7667 spurious n insertions.',
    'Negation should lower CER by about 0.60-0.70 absolute and reduce the length excess.',
    'set_rows',
)
s.edit('set_rows', [name], hypothesis=h, index=[row],
       values=-s.subject.tensors()[name][[row]])
s.keep_unmeasured('Reverse the isolated sign error while preserving the adapted n row and its bias.')
s.freeze()
