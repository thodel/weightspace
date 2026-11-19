"""TrOCR planted-defect control, s07, task (a): budget 10, with the donor. Written in full before the first pass."""
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
NAMES = ["dec.05.cross.o.weight", "dec.05.cross.o.bias"]

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
donor = {n: P[n].copy() for n in NAMES}
del P

T = s.subject.tensors()
zeroed = all(float(abs(T[n]).max()) == 0.0 for n in NAMES)
s.note(f"dec.05.cross.o weight and bias exactly zero in the subject: {zeroed}")

m0 = s.measure("C0", rule_mask=MASK)
s.note(f"C0 CER {m0.cer:.4f}, collapsed {m0.collapsed}")

kept1 = False
if zeroed:
    h1 = s.hypothesis(
        "dec.05.cross.o.{weight,bias} were zeroed; the parent's block restores the layer's cross-attention output",
        "CER falls by at least 0.10 from about 0.232 (to 0.05-0.12); collapsed lines <= 3",
        "transplant")
    s.edit("transplant", NAMES, hypothesis=h1, per_tensor={n: {"donor": donor[n]} for n in NAMES})
    m1 = s.measure("H1 transplant dec.05.cross.o from parent", rule_mask=MASK)
    s.note(f"H1 CER {m1.cer:.4f} (C0 {m0.cer:.4f}), collapsed {m1.collapsed}")
    kept1 = s.keep("parent block for the zeroed dec.05.cross.o")

    if not kept1:
        h1b = s.hypothesis(
            "the full parent block is too strong for the child's layer 5; half of it still restores most of the function",
            "CER falls by 0.03-0.10 against C0",
            "interpolate")
        s.edit("interpolate", NAMES, hypothesis=h1b, alpha=0.5,
               per_tensor={n: {"other": donor[n]} for n in NAMES})
        m1b = s.measure("H1b half parent block in dec.05.cross.o", rule_mask=MASK)
        s.note(f"H1b CER {m1b.cer:.4f}, collapsed {m1b.collapsed}")
        s.keep("half parent block for the zeroed dec.05.cross.o")
    elif m1.cer > 0.08 and s.budget_left >= 1:
        h2 = s.hypothesis(
            "a healthy child's cross.o is slightly larger than the parent's (norm ratio about 1.05 on the bias)",
            "change within noise, |dCER| < 0.005; the keep rule will most likely refuse",
            "scale")
        s.edit("scale", NAMES, hypothesis=h2, factor=1.05)
        m2 = s.measure("H2 scale transplanted dec.05.cross.o by 1.05", rule_mask=MASK)
        s.note(f"H2 CER {m2.cer:.4f}, collapsed {m2.collapsed}")
        s.keep("scaled transplanted block")
else:
    s.note("the tensors are not zero as read from the pack; no edit applied")

s.note("no further defect visible in the readings; remaining passes unspent")
summary = s.freeze()
print(summary)
