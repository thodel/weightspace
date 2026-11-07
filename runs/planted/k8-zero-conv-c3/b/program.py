from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")
donor = KrakenSubject(PARENT).tensors()
names = ["C_3.co.weight", "C_3.co.bias"]
h = s.hypothesis("C_3 conv block is zeroed (dead network, CER 1.0); restore from parent",
                 "CER falls from 1.0 to roughly 0.15-0.5", "transplant")
s.edit("transplant", names, hypothesis=h,
       per_tensor={n: {"donor": donor[n].copy()} for n in names})
s.keep_unmeasured("C_3 is exactly zero; parent donor is the only available estimate")
s.freeze()
