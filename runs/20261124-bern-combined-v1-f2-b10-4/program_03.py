"""program_03 — no pass. Freezes the kept state of program_01 (H1 + H2).

Passes 9 and 10 stay unspent: preregistration.md section 5, round 2.
"""
from wsp.run import open_session

s = open_session(".")
B = "O_18.lin.bias"
print("budget left:", s.budget_left)
print("blank bias of the restored state:", float(s.subject.tensors()[B][0]), "(expected about 0.718)")
print("digest of the restored state:", s.subject.digest()[:12], "(expected 9667acb5dd9d, the state kept after pass 3)")
s.note("Round 2: F1, F2a, F2b and F4 (passes 5-8) were all undone. The round-1 plan allowed a refinement only of a "
       "class kept in passes 5-8; nothing was kept, so passes 9 and 10 stay unspent. Frozen state: H1 (all 28 "
       "trained tensors 0.25 toward hertwig-v2, rows by character) + H2 (blank bias -0.25); practice CER 0.1364 "
       "against 0.1456 at C0, collapsed 21 against 29. Predicted held-out change: -0.003 (range -0.010 to +0.004).")
summary = s.freeze()
print("frozen: used", summary["used"], "of", summary["budget"], "| edits", summary["edits"], "| kept", summary["kept"])
print("start digest", summary["start_digest"][:12], "final digest", summary["final_digest"][:12])
print("baseline CER", round(summary["baseline"]["cer"], 4), "| last pass CER (pass 8, F4, undone)",
      round(summary["last_pass"]["cer"], 4))
print("changed tensors:", len(summary["changed_tensors"]))
