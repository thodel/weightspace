from wsp.run import open_session
import numpy as np

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

NAMES = []
for i in range(12):
    NAMES += [f"enc.{i:02d}.ffn.out.weight", f"enc.{i:02d}.ffn.out.bias"]


def try_scale(label, factor, text, pred):
    h = s.hypothesis(text, pred, "scale")
    s.edit("scale", NAMES, hypothesis=h, factor=factor)
    m = s.measure(label, rule_mask=MASK)
    return m, s.keep(label)


m0 = s.measure("C0", rule_mask=MASK)
base = m0.cer
used = 1

m1, k1 = try_scale(
    "H1 enc ffn.out x1/0.7", 1.0 / 0.7,
    "all encoder ffn.out weight+bias were shrunk to 0.7; undo it",
    "CER down 4 to 7 points, fewer collapsed lines")
used += 1
s.note(f"H1 cer {base:.4f} -> {m1.cer:.4f} collapsed {m0.collapsed}->{m1.collapsed} kept={k1}")

def kept_ok(k):
    return bool(k) if not isinstance(k, dict) else bool(k.get("kept", False))

if kept_ok(k1):
    m2, k2 = try_scale(
        "H2 extra x1.05", 1.05,
        "the true factor may be a little larger than 1/0.7 since the child moved these tensors too",
        "CER change within +-0.3 point")
    if not kept_ok(k2):
        m3, k3 = try_scale(
            "H3 extra x0.95", 0.95,
            "the exact restoring factor may be a little smaller than 1/0.7",
            "CER change within +-0.3 point")
else:
    m4, k4 = try_scale(
        "H4 enc ffn.out x1.2", 1.2,
        "partial restore of the encoder ffn.out shrink",
        "CER down 1 to 3 points if the defect is real")

summary = s.freeze()
