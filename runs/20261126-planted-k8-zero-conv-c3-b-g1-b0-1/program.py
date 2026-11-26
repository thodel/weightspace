"""Blind repair script for s09 task (b) (budget 0).

Restores the zeroed C_3 convolutional layer (C_3.co.weight and C_3.co.bias)
by transplanting from the parent model kraken-german-xix-v1.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")
parent = KrakenSubject(PARENT)
P = parent.tensors()

h1 = s.hypothesis(
    "C_3.co.weight and C_3.co.bias are completely zeroed out (L2 = 0.00), causing 100% line collapse (CER 1.0000); transplanting C_3 from parent kraken-german-xix-v1 restores the feature pipeline.",
    "CER drops from 1.0000 by ~0.92-0.97 to ~0.03-0.08 and collapsed lines drop from 300 to 0.",
    "transplant",
)

s.edit(
    "transplant",
    ["C_3.co.weight", "C_3.co.bias"],
    hypothesis=h1,
    per_tensor={
        "C_3.co.weight": {"donor": P["C_3.co.weight"]},
        "C_3.co.bias": {"donor": P["C_3.co.bias"]},
    },
)
s.keep_unmeasured("C_3.co.weight and C_3.co.bias are all zeros in the subject; transplant from parent restores visual feature flow.")

s.freeze()
