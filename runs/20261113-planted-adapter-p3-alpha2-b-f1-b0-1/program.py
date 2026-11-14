"""q02, task (b), budget 0: the blind programmer's script (see hypotheses.md).

One edit: every lora_B scaled by 0.5, undoing a doubled update (lora_alpha 256 where the
recipe and both siblings have 128). No pass, no measure; guarded on the config and on the
pack's total norm so that nothing is applied to a subject the pack did not describe.
"""
from wsp.run import open_session
import numpy as np

PACK_ALPHA = 256
PACK_R = 64
PACK_MODULES = 128
PACK_TOTAL_NORM = 234.99      # ‖ΔW‖ over all modules, Frobenius, with this adapter's own scale (pack.md, Totals)
TOLERANCE = 0.10
FACTOR = 0.5                  # alpha 256 -> 128 equivalent, written into B (the config is copied unchanged on save)


def total_delta_norm(sub) -> float:
    """sqrt(sum over modules of ‖(alpha/r)·B·A‖_F²) without forming any ΔW:
    ‖BA‖_F² = tr((BᵀB)(AAᵀ)), two r×r matrices per module."""
    T = sub.tensors()
    total = 0.0
    for m in sub.modules():
        b = T[f"{m}.lora_B.weight"].astype(np.float64)
        a = T[f"{m}.lora_A.weight"].astype(np.float64)
        total += float(np.einsum("ij,ji->", b.T @ b, a @ a.T))
    return float(sub.scale) * float(np.sqrt(total))


s = open_session(".")
sub = s.subject
assert s.budget == 0, "task (b) is a budget-0 session"

modules = sub.modules()
alpha = sub.config.get("lora_alpha")
r = sub.config.get("r")
norm_before = total_delta_norm(sub)
s.note(f"readings on open: lora_alpha={alpha}, r={r}, scale={sub.scale}, modules={len(modules)}, "
       f"total ‖ΔW‖={norm_before:.2f} (pack: {PACK_TOTAL_NORM})")

guards = []
if alpha != PACK_ALPHA:
    guards.append(f"lora_alpha is {alpha}, the pack says {PACK_ALPHA}")
if r != PACK_R:
    guards.append(f"r is {r}, the pack says {PACK_R}")
if len(modules) != PACK_MODULES:
    guards.append(f"{len(modules)} modules, the pack says {PACK_MODULES}")
if abs(norm_before - PACK_TOTAL_NORM) > TOLERANCE * PACK_TOTAL_NORM:
    guards.append(f"total ‖ΔW‖ {norm_before:.2f} is not within {TOLERANCE:.0%} of the pack's {PACK_TOTAL_NORM}")

if guards:
    s.note("guard failed, nothing applied: " + "; ".join(guards))
else:
    names_B = [f"{m}.lora_B.weight" for m in modules]
    h = s.hypothesis(
        "H1: the update is doubled. lora_alpha is 256 where the recipe and both siblings have 128 (scale 4.0 vs 2.0); "
        "every one of the 128 modules sits at ~1.78x the sibling median (range 1.60-1.85) with ordinary ‖B‖, ‖A‖, i.e. a "
        "global factor, not a module fault. C0 shows 29/32 pages at the cap, 28 loops, CER 2.50: an over-strong adapter. "
        "Scaling every lora_B by 0.5 is the same as setting lora_alpha back to 128.",
        "CER on the practice set from 2.50 down to roughly 0.05-0.15 (a drop of >= 2 CER points absolute); pages at the "
        "cap from 29 to <= 3; loops essentially gone; length ratio back near 1. If the adapter had really been trained at "
        "256, the cost is a few CER points at most.",
        "global scale (alpha)",
    )
    s.edit("scale", names_B, hypothesis=h, factor=FACTOR)
    norm_after = total_delta_norm(sub)
    s.note(f"after the edit: total ‖ΔW‖={norm_after:.2f} (expected {FACTOR * norm_before:.2f}; siblings 127.35 / 135.65)")
    s.keep_unmeasured(
        "budget 0, unmeasured. Config alpha 256 vs recipe 128, uniform ~1.78x norm ratio over all 128 modules, and a C0 "
        "reading with 29/32 pages at the cap all point to one global doubling of the update; halving B restores the "
        "recipe's alpha-128 strength of this adapter's own trained weights. The wrong-case cost (a mildly weak adapter) "
        "is small next to the right-case gain (a looping adapter made usable)."
    )

s.note("H2 not applied: layers.27.self_attn.q_proj has the lowest q_proj effrank (17.0) and the largest attention σ1 "
       "(16.5), but its norm ratio (1.81) is in line with the rest, so no energy was added; possibly natural for a late "
       "full-attention layer. Not projected out blind.")
s.note("H3 not applied: layers.3.self_attn.q_proj (ratio 1.60) and layers.23.self_attn.q_proj (1.68) are mildly below the "
       "~1.78 mean; within run-to-run variation, nothing to correct blind.")

summary = s.freeze()
print("edits:", summary["edits"], "kept:", summary["kept"], "changed tensors:", len(summary["changed_tensors"]))
