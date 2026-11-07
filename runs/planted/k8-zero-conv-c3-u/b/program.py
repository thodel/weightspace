"""Blind programmer (budget 0). Diagnosis: C_3.co.{weight,bias} are exactly zero, which kills the network (CER 1.0, all lines collapsed).
No compatible healthy source exists (parent unavailable; the donor's C_3 encodes different features and would not match this model's C_6),
so no edit is applied."""
from wsp.run import open_session

s = open_session(".")
T = s.subject.tensors()
zero = [n for n, a in T.items() if n.startswith("C_3") and not a.any()]
s.note("C_3 tensors all-zero: %s. No compatible donor for this block (donor C_3 features are for another hand and "
       "C_6 here expects its own parent's C_3); shipping no edit." % zero)
s.freeze()
