from wsp.run import open_session

s = open_session('.')
s.note('No blind edits: the deletion counts do not isolate a blank-row fault, and the parent differences do not identify a reliable repair.')
s.freeze()
