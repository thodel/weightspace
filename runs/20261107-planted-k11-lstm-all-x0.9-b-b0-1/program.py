import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")
T = s.subject.tensors()
P = KrakenSubject(PARENT).tensors()

blocks = ["L_12", "L_14", "L_16"]
sfx = ["", "_reverse"]
bias = [f"{b}.layer.bias_hh_l0{r}" for b in blocks for r in sfx]
ratios = [float(np.linalg.norm(T[n]) / np.linalg.norm(P[n])) for n in bias]
mean_r = float(np.mean(ratios))
names = [f"{b}.layer.{k}_l0{r}" for b in blocks for r in sfx for k in ("weight_ih", "weight_hh", "bias_hh")]

factor = None
if all(0.87 <= r <= 0.93 for r in ratios):
    factor = 1.0 / 0.9
elif all(1.07 <= r <= 1.13 for r in ratios):
    factor = 1.0 / 1.1

if factor is not None:
    h = s.hypothesis("All LSTM tensors were uniformly rescaled (norm ratio to parent %.3f); undo it" % mean_r,
                     "CER down (~0.073 -> ~0.055), fewer deletions", "scale")
    s.edit("scale", names, hypothesis=h, factor=factor)
    s.keep_unmeasured("bias_hh cosine to parent 0.999 with rel L2 0.11 in all six LSTM biases: uniform scale")
else:
    s.note("LSTM bias norm ratios %s not near 0.9/1.1; nothing applied" % ratios)

summary = s.freeze()
