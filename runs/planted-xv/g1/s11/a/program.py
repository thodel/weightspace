"""Weight-space repair program for pack s11, task (a) (budget 10).

Diagnosis from pack readings:
- L_14's four weight matrices (weight_ih_l0, weight_hh_l0, weight_ih_l0_reverse,
  weight_hh_l0_reverse) show anomalous divergence from the parent kraken-german-xix-v1:
  rel L2 = 0.7520 / 0.7400 on ih (cos 0.786 / 0.792) and 0.5483 / 0.5572 on hh
  (cos 0.874 / 0.871), with L2 norms inflated by ~21% (ih) and ~13% (hh) while
  abs max remains <= 2.063 and L_14.layer.bias_hh_l0[_reverse] moved by only
  0.0327 / 0.0317 rel L2 (cos 0.99951).
- By law of cosines, all four L_14 weight tensors carry isotropic additive noise
  of per-element RMS sigma ~ 0.14.
"""
from __future__ import annotations

import numpy as np
from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

L14_WEIGHTS = [
    "L_14.layer.weight_ih_l0",
    "L_14.layer.weight_hh_l0",
    "L_14.layer.weight_ih_l0_reverse",
    "L_14.layer.weight_hh_l0_reverse",
]
L14_BIASES = [
    "L_14.layer.bias_hh_l0",
    "L_14.layer.bias_hh_l0_reverse",
]
HEALTHY_LSTM_WEIGHTS = [
    "L_12.layer.weight_ih_l0",
    "L_12.layer.weight_hh_l0",
    "L_12.layer.weight_ih_l0_reverse",
    "L_12.layer.weight_hh_l0_reverse",
    "L_16.layer.weight_ih_l0",
    "L_16.layer.weight_hh_l0",
    "L_16.layer.weight_ih_l0_reverse",
    "L_16.layer.weight_hh_l0_reverse",
]


def top_deletions(confusions: list) -> int:
    """Count top-confusion deletions (reference char -> empty hypothesis)."""
    total = 0
    for item in confusions or []:
        if isinstance(item, dict):
            ref, hyp, cnt = item.get("ref"), item.get("hyp"), int(item.get("count", 0))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            ref, hyp, cnt = item[0], item[1], int(item[2])
        else:
            continue
        if ref and not hyp:
            total += cnt
    return total


def decide_edit(s, m, best_cer: float, best_collapsed: int, label: str) -> tuple[bool, float, int]:
    """Check collapsed lines and CER, then delegate to the harness keep rule."""
    dels = top_deletions(m.confusions)
    s.note(
        f"{label}: CER={m.cer:.4f} (best={best_cer:.4f}), WER={m.wer:.4f}, "
        f"collapsed={m.collapsed} (best={best_collapsed}), top_dels={dels}"
    )
    if m.collapsed > best_collapsed + 2:
        s.revert(f"{label}: collapsed lines rose from {best_collapsed} to {m.collapsed}")
        return False, best_cer, best_collapsed
    kept = s.keep(f"{label}: CER {best_cer:.4f} -> {m.cer:.4f}, collapsed {m.collapsed}")
    if kept:
        return True, m.cer, m.collapsed
    return False, best_cer, best_collapsed


def main() -> None:
    s = open_session(".")
    T = s.subject.tensors()
    parent_subj = KrakenSubject(PARENT)
    P = parent_subj.tensors()

    orig_L14 = {n: T[n].copy() for n in L14_WEIGHTS}
    parent_L14 = {n: P[n].copy() for n in L14_WEIGHTS}

    # Pass 1: C0 baseline
    c0 = s.measure("C0")
    best_cer = c0.cer
    best_collapsed = c0.collapsed
    s.note(
        f"C0 baseline: CER={c0.cer:.4f}, WER={c0.wer:.4f}, "
        f"collapsed={c0.collapsed}, top_dels={top_deletions(c0.confusions)}"
    )

    # H1: Transplant the 4 corrupted L_14 weight tensors from parent donor
    h1 = s.hypothesis(
        "L_14's four weight tensors carry additive isotropic noise (sigma ~ 0.14); "
        "transplanting them from the parent donor removes the corruption while "
        "preserving child fine-tuning in all other layers and L_14 biases.",
        "CER decrease by 0.0400 to 0.0700 (from 0.1200 to ~0.0500-0.0800), collapsed lines <= 3",
        "transplant",
    )
    s.edit(
        "transplant",
        L14_WEIGHTS,
        hypothesis=h1,
        per_tensor={n: {"donor": parent_L14[n]} for n in L14_WEIGHTS},
    )
    m1 = s.measure("H1_transplant_L14_weights")
    kept_h1, best_cer, best_collapsed = decide_edit(s, m1, best_cer, best_collapsed, "H1")

    # H2: Transplant L_14 bias_hh_l0 and bias_hh_l0_reverse from parent donor
    h2 = s.hypothesis(
        "Transplanting L_14.layer.bias_hh_l0 and bias_hh_l0_reverse from the parent "
        "donor aligns L_14 recurrent gate biases exactly with the parent L_14 weights.",
        "CER decrease by 0.0005 to 0.0030",
        "transplant",
    )
    s.edit(
        "transplant",
        L14_BIASES,
        hypothesis=h2,
        per_tensor={n: {"donor": P[n].copy()} for n in L14_BIASES},
    )
    m2 = s.measure("H2_transplant_L14_biases")
    kept_h2, best_cer, best_collapsed = decide_edit(s, m2, best_cer, best_collapsed, "H2")

    # H3: Interpolate L_14 weights to 90% parent / 10% original subject
    h3 = s.hypothesis(
        "Blending 90% parent L_14 weights with 10% original subject L_14 weights "
        "recovers part of the child's fine-tuning adaptation while suppressing 90% of the noise.",
        "CER decrease by 0.0005 to 0.0040",
        "interpolate",
    )
    if kept_h1:
        # Current L_14 weights are parent_L14; alpha=0.10 toward orig_L14 yields 0.90*P + 0.10*S
        s.edit(
            "interpolate",
            L14_WEIGHTS,
            hypothesis=h3,
            alpha=0.10,
            per_tensor={n: {"other": orig_L14[n]} for n in L14_WEIGHTS},
        )
    else:
        # Current L_14 weights are orig_L14; alpha=0.90 toward parent_L14 yields 0.90*P + 0.10*S
        s.edit(
            "interpolate",
            L14_WEIGHTS,
            hypothesis=h3,
            alpha=0.90,
            per_tensor={n: {"other": parent_L14[n]} for n in L14_WEIGHTS},
        )
    m3 = s.measure("H3_interp_L14_90parent_10subj")
    kept_h3, best_cer, best_collapsed = decide_edit(s, m3, best_cer, best_collapsed, "H3")

    # H4: Interpolate L_14 weights to 85% parent / 15% original subject (linear MMSE optimum)
    h4 = s.hypothesis(
        "Interpolating L_14 weights to the linear MMSE blend (85% parent, 15% original subject) "
        "minimizes expected Euclidean distance to the uncorrupted child weights.",
        "CER decrease by 0.0005 to 0.0050",
        "interpolate",
    )
    if kept_h3:
        # Current is 0.90*P + 0.10*S; alpha = 0.05/0.90 = 1/18 toward orig_L14 yields 0.85*P + 0.15*S
        s.edit(
            "interpolate",
            L14_WEIGHTS,
            hypothesis=h4,
            alpha=1.0 / 18.0,
            per_tensor={n: {"other": orig_L14[n]} for n in L14_WEIGHTS},
        )
    elif kept_h1:
        # Current is 1.00*P; alpha = 0.15 toward orig_L14 yields 0.85*P + 0.15*S
        s.edit(
            "interpolate",
            L14_WEIGHTS,
            hypothesis=h4,
            alpha=0.15,
            per_tensor={n: {"other": orig_L14[n]} for n in L14_WEIGHTS},
        )
    else:
        # Current is 1.00*S; alpha = 0.85 toward parent_L14 yields 0.85*P + 0.15*S
        s.edit(
            "interpolate",
            L14_WEIGHTS,
            hypothesis=h4,
            alpha=0.85,
            per_tensor={n: {"other": parent_L14[n]} for n in L14_WEIGHTS},
        )
    m4 = s.measure("H4_interp_L14_85parent_15subj")
    kept_h4, best_cer, best_collapsed = decide_edit(s, m4, best_cer, best_collapsed, "H4")

    # H5: Scale L_14 weights by the empirical child/parent L2 norm ratio of healthy BiLSTMs (L_12, L_16)
    T = s.subject.tensors()
    ratios = [
        float(np.linalg.norm(T[n]) / np.linalg.norm(P[n]))
        for n in HEALTHY_LSTM_WEIGHTS
    ]
    mean_ratio = float(np.mean(ratios))
    s.note(f"Empirical L_12/L_16 child-to-parent L2 norm ratio: {mean_ratio:.5f}")
    h5 = s.hypothesis(
        f"Scaling L_14 weights by the empirical child/parent L2 norm ratio of L_12 and L_16 "
        f"({mean_ratio:.4f}) matches the child's recurrent gain into L_16.",
        "CER decrease by 0.0005 to 0.0025",
        "scale",
    )
    s.edit("scale", L14_WEIGHTS, hypothesis=h5, factor=mean_ratio)
    m5 = s.measure("H5_scale_L14_norm_ratio")
    kept_h5, best_cer, best_collapsed = decide_edit(s, m5, best_cer, best_collapsed, "H5")

    # H6: Transplant C_3.co.bias from parent (rel L2 = 1.0871, cos = 0.67929 in diff)
    h6 = s.hypothesis(
        "Transplanting C_3.co.bias from the parent tests whether its high relative L2 change "
        "(1.0871, cos 0.679) reflects a secondary perturbation rather than small-norm drift.",
        "CER decrease by 0.0005 to 0.0020",
        "transplant",
    )
    s.edit("transplant", ["C_3.co.bias"], hypothesis=h6, donor=P["C_3.co.bias"].copy())
    m6 = s.measure("H6_transplant_C3_bias")
    kept_h6, best_cer, best_collapsed = decide_edit(s, m6, best_cer, best_collapsed, "H6")

    # H7: Adjust CTC blank bias O_18.lin.bias[0] by the parent-to-child shift (-0.113)
    T = s.subject.tensors()
    delta_bias = np.zeros_like(T["O_18.lin.bias"])
    blank_shift = float(P["O_18.lin.bias"][0] - T["O_18.lin.bias"][0])
    delta_bias[0] = blank_shift if abs(blank_shift) > 1e-4 else -0.113
    h7 = s.hypothesis(
        f"Adding {float(delta_bias[0]):.4f} to O_18.lin.bias[0] (restoring the parent CTC blank "
        "bias level) tests whether reducing blank dominance recovers deleted characters.",
        "CER decrease by 0.0005 to 0.0025",
        "add",
    )
    s.edit("add", ["O_18.lin.bias"], hypothesis=h7, delta=delta_bias)
    m7 = s.measure("H7_adjust_blank_bias")
    kept_h7, best_cer, best_collapsed = decide_edit(s, m7, best_cer, best_collapsed, "H7")

    s.note(
        f"Session complete: best_cer={best_cer:.4f}, best_collapsed={best_collapsed}, "
        f"kept=(H1:{kept_h1}, H2:{kept_h2}, H3:{kept_h3}, H4:{kept_h4}, "
        f"H5:{kept_h5}, H6:{kept_h6}, H7:{kept_h7})"
    )
    s.freeze()


if __name__ == "__main__":
    main()
