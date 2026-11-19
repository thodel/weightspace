"""TrOCR planted-defect control, s02, task (a): budget 10, with the donor.
Written in full before the first pass. See preregistration_hypotheses.md."""
import numpy as np

from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
NAME = "dec.embed.norm.bias"

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
p_bias = np.array(P[NAME], dtype=np.float32, copy=True)
p_weight = np.array(P["dec.embed.norm.weight"], dtype=np.float32, copy=True)
del P

m0 = s.measure("C0", rule_mask=MASK)
s.note(f"C0 CER {m0.cer:.4f}, collapsed {m0.collapsed}")

d = np.array(s.subject.tensors()[NAME], dtype=np.float64) - p_bias.astype(np.float64)
absd = np.abs(d)
spread = float(absd.std() / max(absd.mean(), 1e-12))
s.note(f"{NAME} minus parent: norm {np.linalg.norm(d):.3f}, mean {d.mean():.4f}, std {d.std():.4f}, "
       f"median |d| {np.median(absd):.4f}, relative spread of |d| {spread:.3f}, share positive {(d > 0).mean():.3f}")

repaired = False
best = m0

if spread < 0.15:
    c = float(np.median(absd))
    planted = (np.sign(d) * c).astype(np.float32)
    h1 = s.hypothesis(
        f"{NAME} carries a planted offset of fixed size {c:.4f} (sign pattern from the difference to the parent); "
        "removing it restores the child's own bias",
        "CER falls from about 0.30 to 0.05-0.12 (18 to 25 points down), collapsed lines 33 -> at most 5",
        "donor repair of a layer-norm vector (add)")
    s.edit("add", [NAME], hypothesis=h1, delta=-planted)
    m1 = s.measure("H1 remove fixed-size offset from dec.embed.norm.bias", rule_mask=MASK)
    repaired = s.keep(f"CER {m0.cer:.4f} -> {m1.cer:.4f}, collapsed {m0.collapsed} -> {m1.collapsed}")
    if repaired:
        best = m1

if not repaired:
    h2 = s.hypothesis(
        f"{NAME} (norm ratio 6.3, cosine 0.12 to the parent) carries added noise that makes the decoder drop text; "
        "the parent's vector is within the healthy distance of the child's true one",
        "CER falls from about 0.30 to 0.05-0.12 (18 to 25 points down), collapsed lines 33 -> at most 5",
        "donor transplant of a layer-norm vector")
    s.edit("transplant", [NAME], hypothesis=h2, donor=p_bias)
    m2 = s.measure("H2 transplant dec.embed.norm.bias from the parent", rule_mask=MASK)
    repaired = s.keep(f"CER {m0.cer:.4f} -> {m2.cer:.4f}, collapsed {m0.collapsed} -> {m2.collapsed}")
    if repaired:
        best = m2
    else:
        s.note("H2 failed: the repair of dec.embed.norm.bias was not kept; the pack shows no other candidate, nothing more is applied")

if repaired and (best.cer > 0.15 or best.collapsed > 8) and s.budget_left >= 1:
    s.note(f"residual after the bias repair: CER {best.cer:.4f}, collapsed {best.collapsed}; top confusions {best.confusions[:5]}")
    h3 = s.hypothesis(
        "a second small defect sits in the same layer norm (dec.embed.norm.weight)",
        "weak: change within +-0.5 CER points, most likely not kept",
        "donor interpolation of a layer-norm vector")
    s.edit("interpolate", ["dec.embed.norm.weight"], hypothesis=h3, other=p_weight, alpha=0.5)
    m3 = s.measure("H3 interpolate dec.embed.norm.weight 0.5 toward the parent", rule_mask=MASK)
    if s.keep(f"CER {best.cer:.4f} -> {m3.cer:.4f}"):
        best = m3
    else:
        s.note("H3 not kept: no further defect is visible in the readings; stop")

s.note(f"final kept state: CER {best.cer:.4f}, collapsed {best.collapsed}, repaired={repaired}, passes left {s.budget_left}")
summary = s.freeze()
