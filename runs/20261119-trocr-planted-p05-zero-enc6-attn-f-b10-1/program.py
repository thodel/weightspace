"""TrOCR planted-defect control, s01, task (a): budget 10, with the donor. Written in full before the first pass."""
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

m0 = s.measure("C0", rule_mask=MASK)
s.note(f"C0: CER {m0.cer:.4f}, collapsed {m0.collapsed}; rule lines {sum(MASK)} of {len(MASK)}")

T = s.subject.tensors()
BLOCK = ["enc.06.attn.o.weight", "enc.06.attn.o.bias"]
zeroed = all(float(abs(T[n]).max()) == 0.0 for n in BLOCK)
s.note(f"enc.06.attn.o weight and bias exactly zero in the loaded model: {zeroed}")

kept = False
if zeroed:
    h1 = s.hypothesis(
        "H1: the attention output projection of encoder layer 6 (weight and bias) was zeroed; the parent's block is close to "
        "the healthy child's (neighbouring layers moved 0.06 to 0.09 from the parent)",
        "CER falls from about 0.33 by at least 0.15 absolute, to 0.05 to 0.15; collapsed lines fall to at most 3",
        "transplant of a block from the donor")
    s.edit("transplant", BLOCK, hypothesis=h1, per_tensor={n: {"donor": P[n]} for n in BLOCK})
    m1 = s.measure("H1 enc.06.attn.o from the parent", rule_mask=MASK)
    kept = s.keep("zeroed block restored from the parent")
    s.note(f"H1: CER {m0.cer:.4f} -> {m1.cer:.4f}, collapsed {m0.collapsed} -> {m1.collapsed}, kept {kept}; "
           f"prediction of a fall of at least 0.15 {'met' if m0.cer - m1.cer >= 0.15 else 'not met'}")

    if not kept:
        LAYER = sorted(n for n in T if n.startswith("enc.06.") and n in P)
        h1b = s.hypothesis(
            "H1b: the parent's output projection alone does not fit the child's layer 6; the whole parent layer does",
            "CER falls by at least 0.10 absolute against C0",
            "transplant of a whole layer from the donor")
        s.edit("transplant", LAYER, hypothesis=h1b, per_tensor={n: {"donor": P[n]} for n in LAYER})
        m2 = s.measure("H1b whole enc.06 from the parent", rule_mask=MASK)
        kept = s.keep("whole encoder layer 6 from the parent")
        s.note(f"H1b: CER {m0.cer:.4f} -> {m2.cer:.4f}, collapsed {m2.collapsed}, kept {kept}")
    elif m1.cer > 0.20:
        s.note("H2: CER still above 0.20 after the repair; no further defect is visible in the readings, nothing else is edited")
else:
    s.note("the zeroed block of the readings is not present in the loaded model; nothing is edited")

s.note(f"end: kept {kept}, passes left {s.budget_left}")
summary = s.freeze()
