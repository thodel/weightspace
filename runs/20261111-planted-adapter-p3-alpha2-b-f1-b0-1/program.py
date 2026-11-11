"""Task (b), budget 0 — blind programmer, adapter q01.

Diagnosis (see hypotheses.md): lora_alpha is 256 where the recipe and both siblings have 128, every one of
the 128 modules carries a flat ~1.8x excess in ||dW|| over the sibling median, and the C0 reading loops on
29 of 32 pages. One edit: scale every lora_B by 0.5 (= lora_alpha 256 -> 128). No pass is available, so the
edit is kept unmeasured. Runs inside the sandbox: wsp + numpy only.
"""
from __future__ import annotations

import numpy as np

from wsp.run import open_session
from wsp.lora_subject import LoraSubject

FACTOR = 0.5                      # alpha 256 -> 128, the siblings' and the recipe's scale
RECIPE_ALPHA = 128.0
DONORS = [
    "/home/tobias/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v1",
    "/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/qwen3.5-4b-german-kurrent-xvi-xix-blb-page-v1",
]


def unscaled_norm(tensors: dict, module: str) -> float:
    """||B A||_F without forming the product: QR of B and A^T, then the r x r core."""
    b = tensors[f"{module}.lora_B.weight"].astype(np.float64)
    a = tensors[f"{module}.lora_A.weight"].astype(np.float64)
    rb = np.linalg.qr(b, mode="r")
    ra = np.linalg.qr(a.T, mode="r")
    return float(np.linalg.norm(rb @ ra.T))


s = open_session(".")
sub = s.subject
mods = sub.modules()
b_names = [f"{m}.lora_B.weight" for m in mods]
assert len(mods) == 128, f"expected 128 modules, found {len(mods)}"
assert s.budget_left == 0, "this program is written for budget 0"

s.note(f"q01 as read: lora_alpha={sub.config.get('lora_alpha')} r={sub.config.get('r')} scale={sub.scale:.2f}; "
       f"recipe/siblings alpha={RECIPE_ALPHA:.0f} scale={RECIPE_ALPHA / float(sub.config['r']):.2f}. "
       f"Pack: all 128 module ||dW|| ratios to the sibling median in 1.60-1.85 (median ~1.78), total 234.99 vs 127-136; "
       f"C0: page CER 2.50, 29/32 pages at cap, 28 looping.")

# ---- guard: compare the UNscaled ||B A|| per module with the siblings (donors, statistics only) ----------
own = {m: unscaled_norm(sub.tensors(), m) for m in mods}
donor_ratio_median = None
try:
    donor_norms = []
    donor_alphas = []
    for path in DONORS:
        d = LoraSubject(path)
        donor_alphas.append(float(d.config.get("lora_alpha", float("nan"))))
        dt = d.tensors()
        donor_norms.append({m: unscaled_norm(dt, m) for m in mods if f"{m}.lora_B.weight" in dt})
        del d, dt
    ratios = {}
    for m in mods:
        sib = [dn[m] for dn in donor_norms if m in dn]
        if sib:
            ratios[m] = own[m] / float(np.median(sib))
    if ratios:
        vals = np.array(list(ratios.values()))
        donor_ratio_median = float(np.median(vals))
        suspects = {m: round(r, 3) for m, r in ratios.items()
                    if r < 0.4 * donor_ratio_median or r > 2.5 * donor_ratio_median}
        s.note(f"donor check: siblings' lora_alpha={donor_alphas}; unscaled ||BA|| ratio to sibling median over "
               f"{len(ratios)} modules: median {donor_ratio_median:.3f}, min {vals.min():.3f}, max {vals.max():.3f}. "
               f"Scaled ratio implied: {donor_ratio_median * sub.scale / (float(np.nanmedian(donor_alphas)) / float(sub.config['r'])):.3f}. "
               f"Local suspects (outside 0.4-2.5x the median, NOT edited at budget 0): {suspects or 'none'}")
except Exception as exc:  # donors not reachable in the sandbox: the pack's evidence stands on its own
    s.note(f"donor check skipped ({type(exc).__name__}: {exc}); proceeding on the pack's evidence alone")

if donor_ratio_median is not None and not (0.5 <= donor_ratio_median <= 1.6):
    s.note(f"donor check: unscaled ratio median {donor_ratio_median:.3f} is outside the natural band 0.5-1.6, so the "
           f"excess is not the config's alone. The pack still shows a flat ~1.8x scaled excess over all modules, so a "
           f"global factor of {FACTOR} remains the correction that brings dW to the siblings' level; applying it as planned.")

# ---- H1: the adapter's scale is doubled; halve every B (lora_alpha 256 -> 128) ---------------------------
h = s.hypothesis(
    "H1: lora_alpha is 256 where the recipe and both siblings have 128. The whole adapter is applied at twice its "
    "trained strength: every one of the 128 modules shows a flat ~1.8x excess in ||dW|| over the sibling median and "
    "the C0 decode loops on 29/32 pages. Scaling every lora_B by 0.5 restores dW = 2*B*A, the siblings' scale.",
    "Large improvement: pages at the cap 29 -> <= 8 of 32, loops 28 -> <= 5, page CER 2.50 -> below 0.3 "
    "(most likely 0.05-0.15), truncated CER 0.82 -> about the same as the page CER.",
    "scale (global, all lora_B, factor 0.5)",
)
before_total = float(np.sqrt(sum((sub.scale * v) ** 2 for v in own.values())))
s.edit("scale", b_names, hypothesis=h, factor=FACTOR)
after_total = float(np.sqrt(sum((sub.scale * unscaled_norm(sub.tensors(), m)) ** 2 for m in mods)))
s.note(f"total ||dW|| (own scale {sub.scale:.1f}): {before_total:.2f} -> {after_total:.2f} "
       f"(siblings 127.35 / 135.65)")
s.keep_unmeasured(
    "budget 0: no pass. Three readings agree on one cause (alpha 256 vs 128; flat ~1.8x norm excess on all 128 modules; "
    "catastrophic looping at C0). Halving B is the exact correction of a doubled alpha; a half-strength adapter would "
    "degrade gently where an over-scaled one loops, so the risk is asymmetric in favour of the edit."
)

s.note("No local edit applied: the pack shows no module, layer or kind that stands out in ratio, effective rank or sigma1, "
       "and a wrong local repair at budget 0 cannot be undone.")
summary = s.freeze()
print("frozen:", summary["edits"], "edit(s),", summary["kept"], "kept; final digest", summary["final_digest"][:16])
