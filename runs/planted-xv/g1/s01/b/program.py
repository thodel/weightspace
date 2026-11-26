"""s01 Task (b): budget-0 blind repair of zeroed C_3 block using the available donor."""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"

s = open_session(".")
donor = KrakenSubject(PARENT)
P = donor.tensors()

s.note(
    "Diagnosis from describe.md and c0_practice.md: C_3.co.weight ([32, 32, 3, 13]) "
    "and C_3.co.bias ([32]) are completely zeroed (L2 = 0.00, abs max = 0.000), "
    "severing forward signal propagation after C_0 and causing 300/300 collapsed lines "
    "(CER 1.0000, length ratio 0.000). All other blocks (C_0, C_6, C_9, L_12..L_16, O_18) "
    "have normal norms. Note: donor kraken-bern-hertwig-v2 is from a different lineage "
    "(near-zero weight cosines in C_0/C_6/C_9/L_12..L_16), so cross-lineage C_3 transplant "
    "only restores non-zero conv signal flow without basis alignment."
)

h1 = s.hypothesis(
    "C_3.co.weight and C_3.co.bias are completely zeroed (L2=0), causing 100% CTC blank "
    "collapse (CER 1.0000). Transplanting C_3.co.weight and C_3.co.bias from the donor "
    "restores non-zero feature propagation between C_0 and C_6.",
    "CER change of 0.00 to -0.15 from 1.0000 (downside bounded by C0 collapse at 1.0000; "
    "recovery limited by cross-lineage channel mismatch)",
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
s.keep_unmeasured(
    "C_3.co.weight and C_3.co.bias are identically zero at C0 (CER 1.0000, 300/300 lines "
    "collapsed); transplanting donor C_3 restores non-zero forward activations."
)

s.freeze()
