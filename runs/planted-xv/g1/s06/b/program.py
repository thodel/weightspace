"""Budget-0 weight-space repair program for pack s06, task (b).

Restores the zeroed-out output layer row for character 'e' in O_18.lin.weight
(and its corresponding bias in O_18.lin.bias) by transplanting the aligned row
from the parent model (kraken-german-xix-v1).
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")
parent = KrakenSubject(PARENT)

cm = s.subject.codec_map()
pm = parent.codec_map()

e_child = cm["e"]
e_parent = pm["e"]

PT = parent.tensors()
w_donor = PT["O_18.lin.weight"][[e_parent]].copy()
b_donor = PT["O_18.lin.bias"][[e_parent]].copy()

h1 = s.hypothesis(
    "Row 'e' in O_18.lin.weight has norm 0.00 (zeroed out; parent norm 7.64, cos 0.000), "
    "causing 1,286+ 'e' deletion/substitution errors on the practice set. Transplanting row 'e' "
    "of O_18.lin.weight and O_18.lin.bias from the parent model restores 'e' emission.",
    "CER decreases by ~0.09 to 0.11 (from 0.1623 to ~0.055-0.070)",
    "set_rows",
)

s.edit(
    "set_rows",
    ["O_18.lin.weight", "O_18.lin.bias"],
    hypothesis=h1,
    index=[e_child],
    per_tensor={
        "O_18.lin.weight": {"values": w_donor},
        "O_18.lin.bias": {"values": b_donor},
    },
)
s.keep_unmeasured("Transplant zeroed-out 'e' output weight row and bias from parent model")

s.freeze()
