import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
W = T["O_18.lin.weight"]
r = cm["e"]
if np.linalg.norm(W[r]) < 1e-6:
    pm = KrakenSubject(PARENT).codec_map()
    donor = KrakenSubject(PARENT).tensors()["O_18.lin.weight"][pm["e"]].copy()
    h = s.hypothesis("Output row of 'e' is zeroed (norm 0.00, cos 0 to parent); 'e' is deleted/confused in 910+ places",
                     "large CER drop, about 0.162 -> 0.05-0.09", "set_rows from parent donor")
    s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h, index=[r], values=donor[None, :])
    s.keep_unmeasured("e row restored from parent by character alignment")
else:
    s.note("e row not zero; nothing applied")
s.freeze()
