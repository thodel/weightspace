"""Weight-space repair program for pack s05, task (a) (budget 10).

Model ID: gemini-4-argon
Diagnosis from pack readings:
- All six non-zero tensors of BiLSTM block L_16 have high cosine similarity with parent
  kraken-german-xix-v1 (bias_hh ~0.999, weight_hh ~0.978, weight_ih ~0.951, matching L_12
  and L_14), but their L2 norms are halved (~50% of L_14 and parent) and their relative L2
  distances to parent are all ~0.505-0.550.
- This 0.5x scaling on L_16 attenuates the final hidden representation feeding into O_18.lin,
  causing the positive-bias CTC blank (+1.232) to dominate over character rows (mean bias -1.921),
  resulting in massive deletion confusions, length ratio 0.828, 24 collapsed lines, and CER 0.1993.
"""
from __future__ import annotations

import numpy as np
from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

L16_NAMES = [
    "L_16.layer.weight_ih_l0",
    "L_16.layer.weight_hh_l0",
    "L_16.layer.bias_hh_l0",
    "L_16.layer.weight_ih_l0_reverse",
    "L_16.layer.weight_hh_l0_reverse",
    "L_16.layer.bias_hh_l0_reverse",
]


def main() -> None:
    s = open_session(".")

    # Pass 1: Baseline C0 measurement
    m0 = s.measure("C0")
    best_cer = m0.cer
    s.note(f"C0 baseline: CER={m0.cer:.4f}, WER={m0.wer:.4f}, collapsed={m0.collapsed}, len_ratio={m0.length_ratio:.3f}")

    # H1: Scale all non-zero L_16 tensors by factor=2.0 to undo the 0.5x scaling defect
    h1 = s.hypothesis(
        "L_16 weights and biases were scaled by 0.5x (cosines with parent 0.947-0.999, rel L2 0.505-0.550, "
        "norms half of L_14/parent), attenuating L_16 outputs so the CTC blank dominates; scaling L_16 by 2.0 restores it.",
        "CER decreases by 0.120 to 0.150 (from 0.1993 to ~0.050-0.075) and collapsed lines drop from 24 to 0-2.",
        "scale",
    )
    s.edit("scale", L16_NAMES, hypothesis=h1, factor=2.0)
    m1 = s.measure("H1_scale_L16_x2")
    kept_h1 = s.keep("Restore L_16 2.0x scale")
    if kept_h1:
        best_cer = m1.cer
    s.note(f"H1 result: CER={m1.cer:.4f}, WER={m1.wer:.4f}, collapsed={m1.collapsed}, kept={kept_h1}")

    # If H1 was not kept or CER remains above 0.075, test fallback hypotheses using parent donor
    if (not kept_h1 or best_cer > 0.075) and s.budget_left >= 1:
        parent = KrakenSubject(PARENT)
        pT = parent.tensors()
        T = s.subject.tensors()

        # H2: Per-tensor norm matching of L_16 to parent
        per_tensor_scale = {}
        for name in L16_NAMES:
            cur_norm = float(np.linalg.norm(T[name]))
            par_norm = float(np.linalg.norm(pT[name]))
            factor = (par_norm / cur_norm) if cur_norm > 0 else 1.0
            per_tensor_scale[name] = {"factor": factor}

        h2 = s.hypothesis(
            "Scale each L_16 tensor to match the exact L2 norm of the parent L_16 tensor.",
            "CER decreases by 0.100 to 0.145 vs C0 (or by 0.005 to 0.030 vs H1).",
            "scale",
        )
        s.edit("scale", L16_NAMES, hypothesis=h2, per_tensor=per_tensor_scale)
        m2 = s.measure("H2_scale_L16_parent_norm")
        kept_h2 = s.keep("Scale L_16 tensors to parent norms")
        if kept_h2:
            best_cer = m2.cer
        s.note(f"H2 result: CER={m2.cer:.4f}, collapsed={m2.collapsed}, kept={kept_h2}")

        # H3: If CER still > 0.080, test interpolating L_16 toward parent (alpha=0.5)
        if best_cer > 0.080 and s.budget_left >= 1:
            per_tensor_interp = {name: {"other": pT[name]} for name in L16_NAMES}
            h3 = s.hypothesis(
                "Interpolate L_16 tensors halfway toward parent kraken-german-xix-v1 (alpha=0.5).",
                "CER decreases by 0.040 to 0.100 relative to current kept state.",
                "interpolate",
            )
            s.edit("interpolate", L16_NAMES, hypothesis=h3, per_tensor=per_tensor_interp, alpha=0.5)
            m3 = s.measure("H3_interp_L16_parent_0.5")
            kept_h3 = s.keep("Interpolate L_16 with parent alpha=0.5")
            if kept_h3:
                best_cer = m3.cer
            s.note(f"H3 result: CER={m3.cer:.4f}, collapsed={m3.collapsed}, kept={kept_h3}")

        # H4: If CER still > 0.075, test transplanting C_3.co.bias from parent
        if best_cer > 0.075 and s.budget_left >= 1:
            h4 = s.hypothesis(
                "Transplant C_3.co.bias from parent (rel L2 1.0871, cos 0.679 vs parent).",
                "CER decreases by 0.005 to 0.020 if C_3.co.bias carries a secondary defect.",
                "transplant",
            )
            s.edit("transplant", ["C_3.co.bias"], hypothesis=h4, donor=pT["C_3.co.bias"])
            m4 = s.measure("H4_transplant_C3_bias")
            kept_h4 = s.keep("Transplant C_3.co.bias from parent")
            if kept_h4:
                best_cer = m4.cer
            s.note(f"H4 result: CER={m4.cer:.4f}, collapsed={m4.collapsed}, kept={kept_h4}")

    s.freeze()


if __name__ == "__main__":
    main()
