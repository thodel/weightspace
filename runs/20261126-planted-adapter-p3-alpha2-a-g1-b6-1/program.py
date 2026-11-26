"""Weight-space session program for pack q01, task (a) (budget 6)."""
from __future__ import annotations

import numpy as np
from wsp.lora_subject import LoraSubject
from wsp.run import open_session

SIBLING_PATHS = [
    "/home/tobias/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v1",
    "/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/qwen3.5-4b-german-kurrent-xvi-xix-blb-page-v1",
]


def module_dw_norm(sub: LoraSubject, mod: str) -> float:
    """Fast Frobenius norm of ΔW = scale * B @ A via 64x64 Gram matrices."""
    t = sub.tensors()
    a = t[f"{mod}.lora_A.weight"].astype(np.float64)
    b = t[f"{mod}.lora_B.weight"].astype(np.float64)
    btb = b.T @ b
    aat = a @ a.T
    sq = float(np.sum(btb * aat))
    return float(sub.scale) * float(np.sqrt(max(sq, 0.0)))


def main() -> None:
    s = open_session(".")
    sub = s.subject
    mods = sub.modules()
    all_b_names = [f"{m}.lora_B.weight" for m in mods]

    # Record initial per-module ||ΔW|| before any edit
    c0_norms = {m: module_dw_norm(sub, m) for m in mods}

    # Load sibling norm statistics
    siblings = [LoraSubject(p) for p in SIBLING_PATHS]
    sib_median_norms = {
        m: float(np.median([module_dw_norm(sib, m) for sib in siblings]))
        for m in mods
    }

    # Pass 1: C0 baseline
    m0 = s.measure("C0")
    s.note(f"C0 baseline: CER={m0.cer:.4f}, collapsed={m0.collapsed}")

    # H1: Global scale of all lora_B by 0.5 (restoring effective lora_alpha from 256 to 128)
    h1 = s.hypothesis(
        "Config lora_alpha is 256 (scale 4.0) vs siblings' 128 (scale 2.0), doubling all 128 module updates "
        "and causing severe repetition looping (29/32 pages at cap in C0). Scaling all lora_B by 0.5 restores "
        "effective lora_alpha=128.",
        "CER decreases by 1.50 to 2.45 (from ~2.50 to ~0.05-1.00) and pages at cap drop below 15.",
        "scale",
    )
    s.edit("scale", all_b_names, hypothesis=h1, factor=0.5)
    m1 = s.measure("H1_global_scale_0.5")
    kept_h1 = s.keep("H1: scale all lora_B by 0.5 to restore effective lora_alpha=128")
    s.note(f"H1 result: CER={m1.cer:.4f}, kept={kept_h1}")

    # H2: Per-module norm match to sibling median ||ΔW||
    # Computed relative to the current live tensor norms so it works whether H1 was kept or refused.
    h2 = s.hypothesis(
        "Per-module ||ΔW|| ratios to sibling median are ~1.79 across all 128 modules (slightly below 2.0). "
        "Scaling each module's lora_B so its ||ΔW|| equals the sibling median (~0.56x of C0, or ~1.12x of H1) "
        "tests whether exact donor-norm calibration improves over uniform 0.5x scaling.",
        "CER decreases by 0.01 to 0.30 vs H1 if H1 was kept (or by 1.50 to 2.45 vs C0 if H1 was refused), "
        "with non-increasing pages at cap.",
        "scale",
    )
    per_tensor_h2 = {}
    for m in mods:
        cur_norm = module_dw_norm(sub, m)
        target_norm = sib_median_norms[m]
        factor = (target_norm / cur_norm) if cur_norm > 0 else 1.0
        per_tensor_h2[f"{m}.lora_B.weight"] = {"factor": float(factor)}
    s.edit("scale", all_b_names, hypothesis=h2, per_tensor=per_tensor_h2)
    m2 = s.measure("H2_sibling_median_norm_match")
    kept_h2 = s.keep("H2: match per-module ||ΔW|| to sibling median")
    s.note(f"H2 result: CER={m2.cer:.4f}, kept={kept_h2}")

    # H3: Global scale to 0.45x of C0 (slightly stronger damping than 0.5x to curb any residual looping)
    h3 = s.hypothesis(
        "Slightly stronger global damping (0.45x of C0 ||ΔW||, equivalent to lora_alpha=115.2) may prevent "
        "remaining borderline pages from entering repetition loops near the token cap.",
        "CER decreases by 0.02 to 0.40 relative to the last kept state, with non-increasing pages at cap.",
        "scale",
    )
    per_tensor_h3 = {}
    for m in mods:
        cur_norm = module_dw_norm(sub, m)
        target_norm = 0.45 * c0_norms[m]
        factor = (target_norm / cur_norm) if cur_norm > 0 else 0.45
        per_tensor_h3[f"{m}.lora_B.weight"] = {"factor": float(factor)}
    s.edit("scale", all_b_names, hypothesis=h3, per_tensor=per_tensor_h3)
    m3 = s.measure("H3_global_scale_0.45_of_C0")
    kept_h3 = s.keep("H3: global scale to 0.45x of C0")
    s.note(f"H3 result: CER={m3.cer:.4f}, kept={kept_h3}")

    # H4: Targeted 0.90x damping of high-sigma1 / low-effective-rank modules
    # (layers.0.mlp.gate_proj, layers.27.self_attn.q_proj, and layers 24-31 mlp.gate_proj)
    target_substrings = (
        "layers.0.mlp.gate_proj",
        "layers.27.self_attn.q_proj",
        "layers.24.mlp.gate_proj",
        "layers.25.mlp.gate_proj",
        "layers.26.mlp.gate_proj",
        "layers.27.mlp.gate_proj",
        "layers.28.mlp.gate_proj",
        "layers.29.mlp.gate_proj",
        "layers.30.mlp.gate_proj",
        "layers.31.mlp.gate_proj",
    )
    h4_b_names = [
        f"{m}.lora_B.weight"
        for m in mods
        if any(m.endswith(sub_name) for sub_name in target_substrings)
    ]
    h4 = s.hypothesis(
        "Modules with the lowest effective rank and largest leading singular value sigma1 "
        "(layers.0.mlp.gate_proj, layers.27.self_attn.q_proj, and layers 24-31 mlp.gate_proj) "
        "may contribute disproportionately to residual logit spikes; damping their lora_B by 0.90x "
        "tests for localized late-layer over-amplification.",
        "CER decreases by 0.01 to 0.20 relative to the last kept state without increasing pages at cap.",
        "scale",
    )
    s.edit("scale", h4_b_names, hypothesis=h4, factor=0.90)
    m4 = s.measure("H4_targeted_high_sigma1_damping")
    kept_h4 = s.keep("H4: damp high-sigma1 / low-effrank gate_proj and l27 q_proj by 0.90")
    s.note(f"H4 result: CER={m4.cer:.4f}, kept={kept_h4}")

    s.freeze()


if __name__ == "__main__":
    main()
