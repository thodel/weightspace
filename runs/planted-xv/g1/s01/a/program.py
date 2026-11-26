"""Weight-space repair program for s01 task (a) (budget 10).

Diagnosis from pack/describe.md and pack/c0_practice.md:
- `C_3.co.weight` ([32, 32, 3, 13]) and `C_3.co.bias` ([32]) have L2 = 0.00,
  rms = 0.0000, abs max = 0.000. The entire C_3 convolutional layer has been
  zeroed out.
- Because C_3 outputs all zeros (followed by ReLU, Dropout, MaxPool), C_6
  receives a constant zero feature map on every line, causing 300/300 lines
  to collapse to empty output (CER = 1.0000).
- Only the unrelated donor `kraken-bern-hertwig-v2.mlmodel` is available.
  We test transplanting C_3 from the donor, refining scale/bias or channel
  alignment with C_0/C_6/C_9, and calibrating the CTC blank bias.
"""
from __future__ import annotations

import numpy as np

from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"


def main() -> None:
    s = open_session(".")
    donor = KrakenSubject(PARENT)
    dT = donor.tensors()

    # Pass 1: Baseline C0 measurement
    m0 = s.measure("C0_baseline")
    s.note(
        f"C0 baseline: CER={m0.cer:.4f}, WER={m0.wer:.4f}, "
        f"collapsed={m0.collapsed}, C_3 weight norm={float(np.linalg.norm(s.subject.tensors()['C_3.co.weight'])):.4f}"
    )

    best_cer = m0.cer

    # Pass 2 (H1): Transplant C_3.co.weight and C_3.co.bias from donor
    h1 = s.hypothesis(
        "C_3.co.weight and C_3.co.bias are identically zero in the subject; "
        "transplanting C_3 from the donor restores feature propagation across layer C_3.",
        "CER decreases by 0.20 to 0.90 (from 1.0000 to <0.80) and collapsed lines drop.",
        "transplant",
    )
    s.edit(
        "transplant",
        ["C_3.co.weight", "C_3.co.bias"],
        hypothesis=h1,
        per_tensor={
            "C_3.co.weight": {"donor": dT["C_3.co.weight"]},
            "C_3.co.bias": {"donor": dT["C_3.co.bias"]},
        },
    )
    m1 = s.measure("H1_transplant_C3_weight_and_bias")
    if m1.cer < best_cer:
        kept1 = s.keep("C_3 transplant from donor reduced CER")
        if kept1:
            best_cer = m1.cer
    else:
        s.revert(f"C_3 weight+bias transplant did not reduce CER ({m1.cer:.4f})")

    # Pass 3 (H2a): Test C_3.co.weight transplant with zero C_3.co.bias
    # (or if H1 was not kept, transplant C_3.co.weight alone)
    T = s.subject.tensors()
    if np.linalg.norm(T["C_3.co.weight"]) > 0:
        h2a = s.hypothesis(
            "Zeroing C_3.co.bias while keeping donor C_3.co.weight avoids donor-specific "
            "channel bias shifts feeding into subject C_6.",
            "CER decreases by 0.005 to 0.05 relative to C_3 weight+bias transplant.",
            "scale",
        )
        s.edit("scale", ["C_3.co.bias"], hypothesis=h2a, factor=0.0)
        m2a = s.measure("H2a_zero_C3_bias")
        if m2a.cer < best_cer:
            if s.keep("Zeroing C_3.co.bias improved CER"):
                best_cer = m2a.cer
        else:
            s.revert("Donor C_3.co.bias works better than zero bias")
    else:
        h2a = s.hypothesis(
            "Transplanting C_3.co.weight alone without donor C_3.co.bias restores signal propagation.",
            "CER decreases by 0.10 to 0.80 from 1.0000.",
            "transplant",
        )
        s.edit("transplant", ["C_3.co.weight"], hypothesis=h2a, donor=dT["C_3.co.weight"])
        m2a = s.measure("H2a_transplant_C3_weight_only")
        if m2a.cer < best_cer:
            if s.keep("C_3.co.weight transplant reduced CER"):
                best_cer = m2a.cer
        else:
            s.revert("C_3.co.weight alone did not reduce CER")

    # Pass 4 (H3a): Test transplanting C_0 + C_3 together from donor
    # Since donor C_3 was trained on donor C_0 outputs, pairing them may improve C_3 activations
    h3a = s.hypothesis(
        "Transplanting C_0.co.weight and C_0.co.bias along with C_3 from the donor aligns "
        "the input feature channels of C_3 with the filters C_3 expects.",
        "CER decreases by 0.02 to 0.30.",
        "transplant",
    )
    s.edit(
        "transplant",
        ["C_0.co.weight", "C_0.co.bias", "C_3.co.weight", "C_3.co.bias"],
        hypothesis=h3a,
        per_tensor={
            "C_0.co.weight": {"donor": dT["C_0.co.weight"]},
            "C_0.co.bias": {"donor": dT["C_0.co.bias"]},
            "C_3.co.weight": {"donor": dT["C_3.co.weight"]},
            "C_3.co.bias": {"donor": dT["C_3.co.bias"]},
        },
    )
    m3a = s.measure("H3a_transplant_C0_and_C3")
    if m3a.cer < best_cer:
        if s.keep("Transplanting C_0 and C_3 together improved CER"):
            best_cer = m3a.cer
    else:
        s.revert(f"Transplanting C_0 with C_3 did not beat current best ({m3a.cer:.4f} vs {best_cer:.4f})")

    # Ensure C_3 is populated if neither H1 nor H3a was kept (e.g. if all lines remained collapsed
    # until scaling or conv block transplant is tried).
    # Pass 5 (H3b): What if all conv layers C_0, C_3, C_6, C_9 from donor are needed, or scaling C_3?
    # Let's check if C_3 is currently zero (meaning H1/H2a/H3a all stayed at CER 1.0) or non-zero.
    T = s.subject.tensors()
    if np.linalg.norm(T["C_3.co.weight"]) == 0:
        # C_3 transplant alone or with C_0 didn't move CER below 1.0; try transplanting C_3 scaled up (x2.0)
        h2b = s.hypothesis(
            "Donor C_3 weights with 2.0x amplitude overcome ReLU/C_6 activation thresholds in the subject.",
            "CER decreases by 0.10 to 0.80 from 1.0000.",
            "transplant",
        )
        s.edit(
            "transplant",
            ["C_3.co.weight", "C_3.co.bias"],
            hypothesis=h2b,
            per_tensor={
                "C_3.co.weight": {"donor": (dT["C_3.co.weight"] * 2.0).astype(np.float32)},
                "C_3.co.bias": {"donor": dT["C_3.co.bias"]},
            },
        )
        m2b = s.measure("H2b_transplant_C3_scaled_2x")
        if m2b.cer < best_cer:
            if s.keep("Scaled 2x C_3 transplant reduced CER"):
                best_cer = m2b.cer
        else:
            s.revert("Scaled 2x C_3 transplant did not reduce CER")
    else:
        # C_3 is active! Test scaling C_3.co.weight by 1.25 to compensate for cross-model feature attenuation
        h2b = s.hypothesis(
            "Scaling transplanted C_3.co.weight by 1.25 compensates for attenuation from mismatched C_0->C_3 channels.",
            "CER decreases by 0.01 to 0.10.",
            "scale",
        )
        s.edit("scale", ["C_3.co.weight"], hypothesis=h2b, factor=1.25)
        m2b = s.measure("H2b_scale_C3_1.25")
        if m2b.cer < best_cer:
            if s.keep("Scaling C_3.co.weight by 1.25 reduced CER"):
                best_cer = m2b.cer
                scaled_up = True
            else:
                scaled_up = False
        else:
            s.revert(f"Scaling C_3 by 1.25 gave CER {m2b.cer:.4f} vs {best_cer:.4f}")
            scaled_up = False

        # Pass 6 (H2c): If 1.25 helped, try 1.20 more (total 1.5x); if 1.25 hurt, try 0.80
        if s.budget_left > 1:
            next_factor = 1.20 if scaled_up else 0.80
            h2c = s.hypothesis(
                f"Scaling C_3.co.weight by {next_factor:.2f} further tunes feature activation magnitude into C_6.",
                "CER decreases by 0.005 to 0.08.",
                "scale",
            )
            s.edit("scale", ["C_3.co.weight"], hypothesis=h2c, factor=next_factor)
            m2c = s.measure(f"H2c_scale_C3_{next_factor:.2f}")
            if m2c.cer < best_cer:
                if s.keep(f"Scaling C_3.co.weight by {next_factor:.2f} reduced CER"):
                    best_cer = m2c.cer
            else:
                s.revert(f"Scaling C_3.co.weight by {next_factor:.2f} did not improve CER")

    # Pass 7 (H4a): Adjust CTC blank bias (O_18.lin.bias[0]) if C_3 is active
    T = s.subject.tensors()
    if s.budget_left > 1 and np.linalg.norm(T["C_3.co.weight"]) > 0:
        delta_b = np.zeros_like(T["O_18.lin.bias"])
        delta_b[0] = -1.0
        h4a = s.hypothesis(
            "Lowering CTC blank bias O_18.lin.bias[0] by 1.0 reduces character deletions caused by "
            "attenuated feature norms after cross-donor C_3 transplant.",
            "CER decreases by 0.005 to 0.05.",
            "add",
        )
        s.edit("add", ["O_18.lin.bias"], hypothesis=h4a, delta=delta_b)
        m4a = s.measure("H4a_blank_bias_minus_1.0")
        if m4a.cer < best_cer:
            if s.keep("Lowering blank bias by 1.0 reduced CER"):
                best_cer = m4a.cer
                blank_lowered = True
            else:
                blank_lowered = False
        else:
            s.revert(f"Lowering blank bias by 1.0 gave CER {m4a.cer:.4f} vs {best_cer:.4f}")
            blank_lowered = False

        # Pass 8 (H4b): If lowering blank bias helped, try another -0.75; otherwise try +0.75
        if s.budget_left > 1:
            step = -0.75 if blank_lowered else 0.75
            delta_b2 = np.zeros_like(s.subject.tensors()["O_18.lin.bias"])
            delta_b2[0] = step
            h4b = s.hypothesis(
                f"Shifting CTC blank bias O_18.lin.bias[0] by {step:+.2f} refines the emission threshold.",
                "CER decreases by 0.003 to 0.03.",
                "add",
            )
            s.edit("add", ["O_18.lin.bias"], hypothesis=h4b, delta=delta_b2)
            m4b = s.measure(f"H4b_blank_bias_{step:+.2f}")
            if m4b.cer < best_cer:
                if s.keep(f"Shifting blank bias by {step:+.2f} reduced CER"):
                    best_cer = m4b.cer
            else:
                s.revert(f"Shifting blank bias by {step:+.2f} did not improve CER")

    s.note(f"Finished repair session with best CER={best_cer:.4f}, budget_left={s.budget_left}")
    s.freeze()


if __name__ == "__main__":
    main()
