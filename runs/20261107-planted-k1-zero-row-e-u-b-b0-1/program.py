# Blind programmer (b): diagnosis is a zeroed output row for 'e'; no safe weight-only repair exists without the parent.
from wsp.run import open_session

s = open_session(".")
s.note("Suspect: O_18.lin.weight row of 'e' is all zero (norm 0.00). No repair applied: parent unavailable, donor row is in another basis.")
summary = s.freeze()
print(summary)
