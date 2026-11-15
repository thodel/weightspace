"""program_03: the last program. No pass, no edit: it freezes the kept state
(H2: 0.75 subject + 0.25 parent on all tensors; H3: blank bias -0.3). Passes 9 and 10 stay unspent.
Runs inside the sandbox; only the wsp API and numpy."""
import sys

import numpy as np

from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"
OUT_B = "O_18.lin.bias"

s = open_session(".")
print("budget left:", s.budget_left)
if s.budget_left != 2:
    print("STOP: expected 2 passes left; nothing frozen")
    sys.exit(0)

# The restored state must be the kept one: blank bias = 0.75 * subject + 0.25 * parent - 0.3.
T = s.subject.tensors()
subj = KrakenSubject(MODELS + "kraken-bern-combined-v1.mlmodel").tensors()
par = KrakenSubject(MODELS + "kraken-bern-hertwig-v2.mlmodel").tensors()
expected = 0.75 * float(subj[OUT_B][0]) + 0.25 * float(par[OUT_B][0]) - 0.3
blank = float(T[OUT_B][0])
conv = "C_9.co.weight"
conv_err = float(np.abs(T[conv] - (0.75 * subj[conv] + 0.25 * par[conv])).max())
print(f"blank bias {blank:+.4f} (expected {expected:+.4f}); max deviation of {conv} from "
      f"0.75 subject + 0.25 parent: {conv_err:.2e}; digest {s.subject.digest()[:16]}")
if abs(blank - expected) > 1e-4 or conv_err > 1e-4:
    print("STOP: the restored state is not the kept state H2 + H3; nothing frozen")
    sys.exit(0)

s.note("Round 2: no further hypothesis. H4-H7 were all undone; H6+H7 together is predicted to be refused "
       "(sum of their per-line changes: -0.28 pt, line interval [-0.62, +0.04], not a measurement) and the "
       "gain of H7 is on lines unreadable at C0. Passes 9 and 10 stay unspent. Freezing the kept state H2 + H3.")
summary = s.freeze()
print("frozen. used", summary["used"], "of", summary["budget"], "| edits", summary["edits"], "| kept", summary["kept"])
print("baseline CER", summary["baseline"]["cer"] if summary["baseline"] else None,
      "| last pass (H7, undone) CER", summary["last_pass"]["cer"] if summary["last_pass"] else None)
print("final digest", summary["final_digest"])
print("changed tensors:", len(summary["changed_tensors"]))
