"""program_04 — last program of session 20261124-bern-combined-v1-f2-b10-3.

No pass. The kept state is frozen as it stands:
  H1  all trained tensors 30 % toward hertwig-v2 (output rows by character), pass 2
  H5  blank bias -0.40, pass 6
Practice CER 0.1359 (C0 0.1456), 20 collapsed (C0 29). Two passes stay unspent, as pre-stated in
preregistration.md section 5, round 2: passes 9-10 were reserved for a second step of an edit kept
in passes 7-8, and neither was kept.
"""
from wsp.run import open_session

s = open_session(".")
print("budget left:", s.budget_left, "; digest before freeze:", s.subject.digest())
print("blank bias:", float(s.subject.tensors()["O_18.lin.bias"][0]))
s.note("Freeze without a further pass: H4 and H5b were refused by the keep rule; the pre-stated plan leaves "
       "passes 9-10 unspent in that case. Frozen state = H1 (30 % toward hertwig-v2, all tensors) + H5 "
       "(blank bias -0.40).")
summary = s.freeze()
print("frozen. used", summary["used"], "of", summary["budget"], "passes; edits", summary["edits"],
      "kept", summary["kept"])
print("final digest:", summary["final_digest"])
print("changed tensors:", len(summary["changed_tensors"]))
