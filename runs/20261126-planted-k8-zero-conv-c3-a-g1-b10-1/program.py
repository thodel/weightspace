"""Weight-space repair program for pack s09, task (a) (budget 10).

Primary diagnosis: `C_3.co.weight` and `C_3.co.bias` are completely zeroed
(L2 = 0.00, rel L2 vs parent = 1.0000), killing all activations at the second
convolutional layer and causing 100% blank collapse (CER 1.0000).
"""
from __future__ import annotations

import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"


def main() -> None:
    s = open_session(".")
    parent = KrakenSubject(PARENT)
    pT = parent.tensors()

    # Pass 1: Baseline C0 measurement
    m0 = s.measure("C0_baseline")
    best_cer = m0.cer
    s.note(f"C0 baseline: CER={m0.cer:.4f}, WER={m0.wer:.4f}, collapsed={m0.collapsed}")

    # H1: Transplant C_3.co.weight and C_3.co.bias from parent
    h1 = s.hypothesis(
        "C_3.co.weight and C_3.co.bias are all zeros in the subject, blocking all forward signal. "
        "Transplanting C_3 from the parent model will restore the convolutional backbone.",
        "CER drops from 1.0000 to < 0.1500 (reduction of ~0.85 to 0.96 CER points) and uncollapses lines.",
        "transplant",
    )
    s.edit(
        "transplant",
        ["C_3.co.weight", "C_3.co.bias"],
        hypothesis=h1,
        per_tensor={
            "C_3.co.weight": {"donor": pT["C_3.co.weight"]},
            "C_3.co.bias": {"donor": pT["C_3.co.bias"]},
        },
    )
    m1 = s.measure("H1_transplant_C3")
    if m1.cer < best_cer and m1.collapsed <= m0.collapsed:
        if s.keep("Transplanting parent C_3 restored forward signal and drastically reduced CER"):
            best_cer = m1.cer
    else:
        s.revert("C_3 transplant did not improve CER")

    s.note(f"After H1: CER={m1.cer:.4f}, WER={m1.wer:.4f}, collapsed={m1.collapsed}")

    # H2: Scale transplanted C_3 to match child/parent norm ratio of neighboring conv layers (C_0 and C_6)
    T = s.subject.tensors()
    ratio_c0 = float(np.linalg.norm(T["C_0.co.weight"]) / np.linalg.norm(pT["C_0.co.weight"]))
    ratio_c6 = float(np.linalg.norm(T["C_6.co.weight"]) / np.linalg.norm(pT["C_6.co.weight"]))
    scale_factor = float(np.sqrt(ratio_c0 * ratio_c6))
    # If the geometric mean ratio is very close to 1.0, test a modest +5% gain adjustment
    if abs(scale_factor - 1.0) < 0.01:
        scale_factor = 1.05

    h2 = s.hypothesis(
        f"Scaling transplanted C_3 by factor {scale_factor:.4f} (derived from neighboring conv norm ratios "
        f"C_0={ratio_c0:.4f}, C_6={ratio_c6:.4f}) to match activation scale expected by fine-tuned C_6.",
        "CER decreases by 0.001 to 0.005 points.",
        "scale",
    )
    s.edit(
        "scale",
        ["C_3.co.weight", "C_3.co.bias"],
        hypothesis=h2,
        factor=scale_factor,
    )
    m2 = s.measure("H2_scale_C3")
    if m2.cer < best_cer and m2.collapsed == 0:
        if s.keep(f"Scaling C_3 by {scale_factor:.4f} significantly reduced CER"):
            best_cer = m2.cer
    else:
        s.revert(f"Scaling C_3 by {scale_factor:.4f} did not improve CER over raw parent transplant")

    # H3: Check whether L_16.layer.weight_ih_l0 abs max (4.702) is an outlier diff vs parent
    T = s.subject.tensors()
    w_l16 = T["L_16.layer.weight_ih_l0"]
    pw_l16 = pT["L_16.layer.weight_ih_l0"]
    diff_l16 = np.abs(w_l16 - pw_l16)
    max_diff_l16 = float(np.max(diff_l16))
    s.note(
        f"L_16.layer.weight_ih_l0 abs max={float(np.max(np.abs(w_l16))):.3f}, "
        f"parent abs max={float(np.max(np.abs(pw_l16))):.3f}, max |child-parent|={max_diff_l16:.3f}"
    )

    if max_diff_l16 > 1.5:
        outlier_rows = [int(r) for r in np.where(np.max(diff_l16, axis=1) > 1.5)[0]]
        if outlier_rows:
            h3 = s.hypothesis(
                f"L_16.layer.weight_ih_l0 has {len(outlier_rows)} row(s) with |child - parent| > 1.5 "
                f"(max diff {max_diff_l16:.3f}); replacing those rows with parent rows removes potential spike.",
                "CER decreases by 0.001 to 0.010 points if spike was harmful.",
                "set_rows",
            )
            s.edit(
                "set_rows",
                ["L_16.layer.weight_ih_l0"],
                hypothesis=h3,
                index=outlier_rows,
                values=pw_l16[outlier_rows],
            )
            m3 = s.measure("H3_L16_outlier_rows")
            if m3.cer < best_cer:
                if s.keep("Replacing outlier rows in L_16.layer.weight_ih_l0 improved CER"):
                    best_cer = m3.cer
            else:
                s.revert("Outlier rows in L_16.layer.weight_ih_l0 are benign fine-tuning weights")

    # H4: Mild interpolation of C_6 toward parent (alpha=0.10) to ease the C_3 (parent) -> C_6 (child) interface
    h4 = s.hypothesis(
        "Since C_3 comes from the parent model while C_6 was fine-tuned with the child's C_3, "
        "interpolating C_6 10% toward the parent (alpha=0.10) may reduce feature mismatch at the C_3->C_6 interface.",
        "CER decreases by 0.001 to 0.005 points.",
        "interpolate",
    )
    s.edit(
        "interpolate",
        ["C_6.co.weight", "C_6.co.bias"],
        hypothesis=h4,
        alpha=0.10,
        per_tensor={
            "C_6.co.weight": {"other": pT["C_6.co.weight"]},
            "C_6.co.bias": {"other": pT["C_6.co.bias"]},
        },
    )
    m4 = s.measure("H4_interp_C6_0.10")
    if m4.cer < best_cer:
        if s.keep("Interpolating C_6 10% toward parent reduced C_3->C_6 interface mismatch"):
            best_cer = m4.cer
    else:
        s.revert("Child C_6 works best unblended with parent C_3")

    s.freeze()


if __name__ == "__main__":
    main()
