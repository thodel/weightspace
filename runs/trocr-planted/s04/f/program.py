"""TrOCR planted-defect control, s04, task (a): budget 10, with the donor. Written in full before the first pass."""
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

W = ["enc.%02d.ffn.out.weight" % i for i in range(12)]
B = ["enc.%02d.ffn.out.bias" % i for i in range(12)]
NAMES = W + B

m0 = s.measure("C0", rule_mask=MASK)
s.note("C0 CER %.4f, collapsed %d" % (m0.cer, m0.collapsed))

# H1: all encoder FFN output projections were multiplied by 0.7 -> undo with 1/0.7
h1 = s.hypothesis(
    "enc.00..11.ffn.out.{weight,bias} were multiplied by 0.7 (norm ratio 0.700, cosine ~1 to the parent); scale back by 1/0.7",
    "CER falls by at least 0.10 (expected about 0.17, to 0.05-0.12); collapsed lines to 0-2; space deletions more than halved",
    "scale")
s.edit("scale", NAMES, hypothesis=h1, factor=1.0 / 0.7)
m1 = s.measure("H1 enc ffn.out x1/0.7", rule_mask=MASK)
kept = s.keep("undo the 0.7 scaling of the encoder FFN output projections")
s.note("H1 kept=%s CER %.4f -> %.4f, collapsed %d -> %d" % (kept, m0.cer, m1.cer, m0.collapsed, m1.collapsed))

if not kept:
    # H2: per-tensor factor from the donor norms
    P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
    T = s.subject.tensors()
    per = {}
    for n in NAMES:
        den = float(np.linalg.norm(T[n]))
        per[n] = {"factor": float(np.linalg.norm(P[n])) / den if den > 0 else 1.0}
    del P
    h2 = s.hypothesis(
        "same family, per-tensor factor = parent norm / subject norm (about 1.42-1.43)",
        "CER falls by at least 0.10",
        "scale")
    s.edit("scale", NAMES, hypothesis=h2, per_tensor=per)
    m2 = s.measure("H2 enc ffn.out to parent norm", rule_mask=MASK)
    kept = s.keep("restore the parent norm of the encoder FFN output projections")
    s.note("H2 kept=%s CER %.4f" % (kept, m2.cer))

    if not kept:
        # H3: weights only
        h3 = s.hypothesis(
            "only the 12 enc.NN.ffn.out.weight scaled by 1/0.7, biases left",
            "CER falls by at least 0.05",
            "scale")
        s.edit("scale", W, hypothesis=h3, factor=1.0 / 0.7)
        m3 = s.measure("H3 enc ffn.out.weight x1/0.7", rule_mask=MASK)
        kept = s.keep("undo the 0.7 scaling of the encoder FFN output weights only")
        s.note("H3 kept=%s CER %.4f" % (kept, m3.cer))
        if not kept:
            s.note("No rescaling of the encoder FFN output family was kept; model left as it is.")
else:
    if m1.cer > 0.15:
        s.note("CER after H1 still above 0.15: a second defect is likely, but no reading points to it; nothing more applied.")
    else:
        s.note("H0: no other tensor is outside the healthy range of the diff; no further edit. Unspent passes left unspent.")

summary = s.freeze()
