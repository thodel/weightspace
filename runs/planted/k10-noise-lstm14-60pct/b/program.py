from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
ALPHA = 0.75

s = open_session(".")
P = KrakenSubject(PARENT).tensors()
names = [f"L_14.layer.{w}{d}" for w in ("weight_ih_l0", "weight_hh_l0") for d in ("", "_reverse")]
ok = all(n in P and P[n].shape == s.subject.tensors()[n].shape for n in names)
if ok:
    h = s.hypothesis("L_14 weights carry additive noise (moved 2-3x more than L_12/L_16 from the parent)",
                     "CER 0.12 -> ~0.07; fewer deletions and collapsed lines", "interpolate toward parent")
    s.edit("interpolate", names, hypothesis=h, alpha=ALPHA,
           per_tensor={n: {"other": P[n].copy()} for n in names})
    s.keep_unmeasured("L_14 noise: interpolate 75% toward parent donor (budget 0)")
else:
    s.note("parent tensors missing or mismatched; nothing applied")
s.freeze()
