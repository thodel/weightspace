"""TrOCR planted-defect control, s08, task (a): budget 10, with the donor.
Written in full before the first pass. Hypotheses: preregistration_hypotheses.md."""
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
T = s.subject.tensors()
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()

# Healthy relative distance of each matrix type: mean of dec.07 and dec.09 in the pack's diff.
HEALTHY_REL = {
    "cross.k": 0.098, "cross.o": 0.113, "cross.q": 0.108, "cross.v": 0.097,
    "ffn.in": 0.145, "ffn.out": 0.189,
    "self.k": 0.082, "self.o": 0.140, "self.q": 0.080, "self.v": 0.111,
}
NAMES = ["dec.08.%s.weight" % k for k in HEALTHY_REL]

W0 = {n: np.array(T[n], dtype=np.float64, copy=True) for n in NAMES}   # the defective originals
P0 = {n: np.array(P[n], dtype=np.float64, copy=True) for n in NAMES}

alpha, sigma = {}, {}
for k, n in zip(HEALTHY_REL, NAMES):
    d = W0[n] - P0[n]
    total = float((d ** 2).sum())
    healthy = (HEALTHY_REL[k] * float(np.linalg.norm(P0[n]))) ** 2
    noise = max(total - healthy, 0.0)
    alpha[n] = float(min(max(noise / total, 0.0), 1.0)) if total > 0 else 0.0
    sigma[n] = float(np.sqrt(noise / d.size))
s.note("Wiener alpha per tensor: %s; sigma estimate: %s" % (
    {n: round(a, 3) for n, a in alpha.items()}, {n: round(v, 4) for n, v in sigma.items()}))


def denoised(n):
    """parent + Gavish-Donoho optimally shrunk SVD of (W_orig - parent)."""
    d = W0[n] - P0[n]
    sg = sigma[n]
    if sg <= 0:
        return W0[n].astype(np.float32)
    u, sv, vt = np.linalg.svd(d, full_matrices=False)
    lo, hi = min(d.shape), max(d.shape)
    beta = lo / hi
    y = sv / (sg * np.sqrt(hi))
    inner = (y ** 2 - beta - 1.0) ** 2 - 4.0 * beta
    eta = np.where(y > 1.0 + np.sqrt(beta), np.sqrt(np.maximum(inner, 0.0)) / np.maximum(y, 1e-12), 0.0)
    sv2 = eta * sg * np.sqrt(hi)
    s.note("%s: %d of %d singular directions kept" % (n, int((sv2 > 0).sum()), lo))
    return (P0[n] + (u * sv2) @ vt).astype(np.float32)


# Pass 1: baseline
m0 = s.measure("C0", rule_mask=MASK)
s.note("C0 cer %.4f collapsed %s" % (m0.cer, m0.collapsed))

kept_any = False
kept_cers = []

# H1: Wiener interpolation toward the parent (pass 2)
h1 = s.hypothesis(
    "dec.08's 10 weight matrices carry additive i.i.d. noise (sigma ~0.020); shrink each toward the parent with a per-tensor Wiener weight (~0.93-0.96)",
    "CER falls by at least 5 points (0.232 -> 0.10-0.17), collapsed lines to 0-1",
    "donor interpolation")
s.edit("interpolate", NAMES, hypothesis=h1,
       per_tensor={n: {"other": P[n], "alpha": alpha[n]} for n in NAMES})
m1 = s.measure("H1 dec.08 matrices Wiener-interpolated to parent", rule_mask=MASK)
k1 = s.keep("H1: noise on dec.08 matrices removed by shrinkage toward the parent")
kept_any = kept_any or k1
if k1:
    kept_cers.append(m1.cer)
s.note("H1 cer %.4f collapsed %s kept %s" % (m1.cer, m1.collapsed, k1))

# H2: spectral denoising of the delta (pass 3); judged against the last kept state
h2 = s.hypothesis(
    "the healthy child delta of dec.08 is partly low-rank; parent + optimally shrunk SVD of the original delta keeps more of it than uniform shrinkage",
    "0 to -1.0 CER points against the H1 state; even odds of clearing the keep rule" if k1
    else "CER falls by at least 3 points against C0",
    "donor + spectral denoising")
T = s.subject.tensors()
s.edit("transplant", NAMES, hypothesis=h2, per_tensor={n: {"donor": denoised(n)} for n in NAMES})
m2 = s.measure("H2 dec.08 matrices = parent + shrunk SVD of delta", rule_mask=MASK)
k2 = s.keep("H2: spectral denoising of the dec.08 delta")
kept_any = kept_any or k2
if k2:
    kept_cers.append(m2.cer)
s.note("H2 cer %.4f collapsed %s kept %s" % (m2.cer, m2.collapsed, k2))

# H3: only if nothing was kept: plain transplant of the 10 matrices (pass 4)
if not kept_any:
    h3 = s.hypothesis(
        "fallback: full transplant of dec.08's 10 weight matrices from the parent",
        "if H1 and H2 failed this is expected to fail too (within 0.5 points of H1); a keep would mean my alpha estimate was wrong",
        "donor transplant")
    s.edit("transplant", NAMES, hypothesis=h3, per_tensor={n: {"donor": P[n]} for n in NAMES})
    m3 = s.measure("H3 dec.08 matrices transplanted", rule_mask=MASK)
    k3 = s.keep("H3: full transplant of dec.08 matrices")
    kept_any = kept_any or k3
    if k3:
        kept_cers.append(m3.cer)
    s.note("H3 cer %.4f collapsed %s kept %s" % (m3.cer, m3.collapsed, k3))

# H4: no further edit; the readings show nothing else out of the healthy range.
best = kept_cers[-1] if kept_cers else m0.cer
s.note("H4: no other tensor is outside the healthy range of the diff; no further edits. "
       "Kept any: %s. Lowest kept-state CER %.4f%s" % (
           kept_any, best, " (still above 0.20: unexplained residual)" if best > 0.20 else ""))
summary = s.freeze()
