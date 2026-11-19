"""TrOCR planted-defect control, s06, task (a): budget 10, with the donor.

Diagnosis: about 20 % Gaussian noise on every 2-D weight matrix of the encoder and decoder layers.
Repair: per-tensor shrinkage interpolation toward the parent, alpha = sigma^2 / d^2 (clipped to 1).
"""
import re

import numpy as np

from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
SIGMA = 0.20

s = open_session(".")
T = s.subject.tensors()
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
P = {n: np.array(v, dtype=np.float32, copy=True) for n, v in P.items()}
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

ENC_RE = re.compile(r"^enc\.\d\d\.(attn\.[qkvo]|ffn\.(in|out))\.weight$")
DATT_RE = re.compile(r"^dec\.\d\d\.(self|cross)\.[qkvo]\.weight$")
DFFN_RE = re.compile(r"^dec\.\d\d\.ffn\.(in|out)\.weight$")


def pick(rx):
    return sorted(n for n in T if rx.match(n) and n in P and T[n].ndim == 2 and T[n].shape == P[n].shape)


ENC, DATT, DFFN = pick(ENC_RE), pick(DATT_RE), pick(DFFN_RE)


def wiener_alpha(names, factor=1.0):
    """alpha = sigma^2 / d^2 with d the relative distance to the parent, from the tensors as they are at C0."""
    out = {}
    for n in names:
        a = np.asarray(T[n], dtype=np.float64)
        b = np.asarray(P[n], dtype=np.float64)
        d = np.linalg.norm(a - b) / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + 1e-12)
        al = 1.0 if d <= SIGMA else (SIGMA / d) ** 2
        out[n] = float(min(1.0, max(0.0, al)) * factor)
    return out


A_ENC, A_DATT, A_DFFN = wiener_alpha(ENC), wiener_alpha(DATT), wiener_alpha(DFFN)


def mean(d):
    return float(np.mean(list(d.values()))) if d else float("nan")


s.note(f"tensors: enc {len(ENC)}, dec attn {len(DATT)}, dec ffn {len(DFFN)}; mean shrinkage alpha "
       f"enc {mean(A_ENC):.3f}, dec attn {mean(A_DATT):.3f}, dec ffn {mean(A_DFFN):.3f} (sigma {SIGMA})")

m0 = s.measure("C0", rule_mask=MASK)
state = {"cer": m0.cer}
s.note(f"C0 CER {m0.cer:.4f}, collapsed {m0.collapsed}")


def trial(label, text, prediction, names, alphas, edit_class="donor interpolation (denoise)"):
    """One hypothesis, one edit, one pass, decided by the keep rule. Returns True if kept."""
    if not names or s.budget_left < 1:
        s.note(f"{label}: skipped (names {len(names)}, passes left {s.budget_left})")
        return False
    h = s.hypothesis(text, prediction, edit_class)
    s.edit("interpolate", names, hypothesis=h,
           per_tensor={n: {"other": P[n], "alpha": float(alphas[n])} for n in names})
    m = s.measure(label, rule_mask=MASK)
    kept = s.keep(f"{label}: CER {state['cer']:.4f} -> {m.cer:.4f}, collapsed {m.collapsed}")
    s.note(f"{label}: CER {m.cer:.4f} (before {state['cer']:.4f}), collapsed {m.collapsed}, kept {kept}")
    if kept:
        state["cer"] = m.cer
    return kept


def scaled(alphas, f):
    return {n: float(a * f) for n, a in alphas.items()}


# H1: encoder matrices
enc_full = False
k1 = trial("H1 enc shrink",
           "All encoder layer weight matrices carry ~20 % added noise (distance 0.20-0.22, norm ratio 1.02, biases unmoved); "
           "shrink them toward the parent with alpha = sigma^2/d^2",
           "CER falls by 6 to 12 points from C0 0.2426; fewer collapsed lines", ENC, A_ENC)
if not k1:
    k1 = trial("H1b enc transplant",
               "Fallback: the true child encoder equals the parent up to ~1 %; transplant the encoder weight matrices",
               "CER falls by 5 to 12 points", ENC, {n: 1.0 for n in ENC}, "donor transplant (denoise)")
    enc_full = k1

# H2: decoder attention matrices
k2 = trial("H2 dec attn shrink",
           "Decoder self/cross attention matrices carry the same noise over a true movement of ~0.07-0.11; shrink toward the parent",
           "CER falls by a further 2 to 6 points", DATT, A_DATT)
datt_f = 1.0
if not k2:
    k2 = trial("H2b dec attn half shrink",
               "Fallback: half the shrinkage on the decoder attention matrices",
               "CER falls by 1 to 4 points", DATT, scaled(A_DATT, 0.5))
    datt_f = 0.5

# H3: decoder FFN matrices
k3 = trial("H3 dec ffn shrink",
           "Decoder FFN matrices carry the same noise over a true movement of ~0.15-0.23; shrink toward the parent (alpha ~0.45-0.65)",
           "CER falls by a further 1 to 4 points (least certain)", DFFN, A_DFFN)
if not k3:
    k3 = trial("H3b dec ffn half shrink",
               "Fallback: half the shrinkage on the decoder FFN matrices",
               "CER falls by 0.5 to 2.5 points", DFFN, scaled(A_DFFN, 0.5))

# H4: rest of the way for the encoder (only if kept with alpha < 1 and it leaves a pass for H5 or is the last use)
if k1 and not enc_full and s.budget_left >= 2:
    trial("H4 enc to parent",
          "Check of sigma: move the already shrunk encoder matrices fully onto the parent",
          "within +-0.5 points; probably not kept", ENC, {n: 1.0 for n in ENC}, "donor transplant (denoise)")

# H5: a further step of the decoder toward the parent
if (k2 or k3) and s.budget_left >= 1:
    names5 = (DATT if k2 else []) + (DFFN if k3 else [])
    trial("H5 dec further 0.3",
          "Check of the shrinkage strength: a further 0.3 step of the kept decoder matrices toward the parent",
          "within +-1 point; probably not kept", names5, {n: 0.3 for n in names5})

s.note(f"final kept CER {state['cer']:.4f} (C0 {m0.cer:.4f}); kept H1 {k1}, H2 {k2}, H3 {k3}; passes left {s.budget_left}")
summary = s.freeze()
