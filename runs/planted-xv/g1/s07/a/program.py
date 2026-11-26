"""Weight-space repair program for pack s07, task (a) (budget 10, unrelated donor)."""
from __future__ import annotations

import numpy as np
from wsp.align import codec_row_map
from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"


def deletion_count(m) -> int:
    """Sum of deletion confusions (ref -> '') in the top confusions list."""
    total = 0
    for item in (m.confusions or []):
        if isinstance(item, dict):
            if item.get("hyp", item.get("hypothesis")) in ("", "∅"):
                total += int(item.get("count", 0))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            if item[1] in ("", "∅"):
                total += int(item[2])
    return total


def decide(s, m, kept_m, label: str) -> tuple[bool, object]:
    """Keep if collapsed lines did not increase and the session's paired 95% CI rule is met."""
    if m.collapsed > kept_m.collapsed:
        s.revert(f"{label}: collapsed lines rose ({kept_m.collapsed} -> {m.collapsed})")
        return False, kept_m
    kept = s.keep(f"{label}: CER {kept_m.cer:.4f} -> {m.cer:.4f}, collapsed {kept_m.collapsed} -> {m.collapsed}")
    if kept:
        return True, m
    return False, kept_m


def main() -> None:
    s = open_session(".")
    donor = KrakenSubject(PARENT)
    donor_T = donor.tensors()
    cm = s.subject.codec_map()
    pm = donor.codec_map()
    row_map = codec_row_map(cm, pm)

    # Pass 1: Baseline C0
    m0 = s.measure("C0")
    kept_m = m0
    s.note(
        f"C0 baseline: CER={m0.cer:.4f}, WER={m0.wer:.4f}, "
        f"collapsed={m0.collapsed}, top_del={deletion_count(m0)}"
    )

    top_del_chars = [" ", "e", "n", "i", "r", "c", "s", "u", "t", "m"]
    top_del_rows = sorted({cm[c] for c in top_del_chars if c in cm and cm[c] != 0})

    # Pass 2: H1 — Subtract 0.50 from CTC blank bias O_18.lin.bias[0]
    T = s.subject.tensors()
    n_classes = T["O_18.lin.bias"].shape[0]
    h1 = s.hypothesis(
        "H1: Subtract 0.50 from CTC blank bias O_18.lin.bias[0] (+1.232 -> +0.732) to reduce blank dominance and character deletions",
        "CER decrease by 0.0020 to 0.0150, fewer deletions and collapsed lines",
        "add",
    )
    d1 = np.zeros(n_classes, dtype=np.float32)
    d1[0] = -0.50
    s.edit("add", ["O_18.lin.bias"], hypothesis=h1, delta=d1)
    m1 = s.measure("H1_blank_bias_m0.50")
    kept_h1, kept_m = decide(s, m1, kept_m, "H1")

    # Pass 3: H2 — Branch on H1 for second blank bias step
    if kept_h1:
        step2 = -0.50
        desc2 = "H2: Subtract an additional 0.50 from O_18.lin.bias[0] (cumulative -1.00) after H1 succeeded"
    else:
        step2 = -0.25
        desc2 = "H2: Subtract a gentler 0.25 from O_18.lin.bias[0] after -0.50 was not kept"
    h2 = s.hypothesis(
        desc2,
        "CER decrease by 0.0015 to 0.0100 vs last kept state",
        "add",
    )
    d2 = np.zeros(n_classes, dtype=np.float32)
    d2[0] = step2
    s.edit("add", ["O_18.lin.bias"], hypothesis=h2, delta=d2)
    m2 = s.measure(f"H2_blank_bias_{step2:+.2f}")
    kept_h2, kept_m = decide(s, m2, kept_m, "H2")

    # Pass 4: H3 — Project out the blank row direction from all non-blank rows of O_18.lin.weight
    T = s.subject.tensors()
    blank_dir = T["O_18.lin.weight"][0].copy()
    non_blank_rows = list(range(1, n_classes))
    h3 = s.hypothesis(
        "H3: Project out the CTC blank weight direction O_18.lin.weight[0] from all non-blank output rows (1..296)",
        "CER decrease by 0.0020 to 0.0200 and fewer collapsed lines",
        "project_out",
    )
    s.edit(
        "project_out",
        ["O_18.lin.weight"],
        hypothesis=h3,
        direction=blank_dir,
        axis=0,
        index=non_blank_rows,
    )
    m3 = s.measure("H3_project_out_blank_all")
    kept_h3, kept_m = decide(s, m3, kept_m, "H3")

    # Pass 5: H4 — Scale down CTC blank weight row O_18.lin.weight[0] by 0.85
    T = s.subject.tensors()
    h4 = s.hypothesis(
        "H4: Scale CTC blank weight row O_18.lin.weight[0] by 0.85 to dampen frame-level blank logit spikes",
        "CER decrease by 0.0010 to 0.0100",
        "set_rows",
    )
    blank_row_scaled = (T["O_18.lin.weight"][0:1] * 0.85).astype(np.float32)
    s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h4, index=[0], values=blank_row_scaled)
    m4 = s.measure("H4_scale_blank_row_0.85")
    kept_h4, kept_m = decide(s, m4, kept_m, "H4")

    # Pass 6: H5 — Add +0.20 to O_18.lin.bias for top-10 deleted characters
    h5 = s.hypothesis(
        "H5: Add +0.20 to O_18.lin.bias for the top-10 deleted characters (' ', 'e', 'n', 'i', 'r', 'c', 's', 'u', 't', 'm')",
        "CER decrease by 0.0010 to 0.0080",
        "add",
    )
    d5 = np.zeros(n_classes, dtype=np.float32)
    for r in top_del_rows:
        d5[r] = 0.20
    s.edit("add", ["O_18.lin.bias"], hypothesis=h5, delta=d5)
    m5 = s.measure("H5_top10_del_bias_p0.20")
    kept_h5, kept_m = decide(s, m5, kept_m, "H5")

    # Pass 7: H6 — Scale O_18.lin.weight rows of the top-10 deleted characters by 1.08
    T = s.subject.tensors()
    h6 = s.hypothesis(
        "H6: Scale O_18.lin.weight rows of the top-10 deleted characters by 1.08 to counteract their weak row norms",
        "CER decrease by 0.0010 to 0.0080",
        "set_rows",
    )
    scaled_top_rows = (T["O_18.lin.weight"][np.asarray(top_del_rows)] * 1.08).astype(np.float32)
    s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h6, index=top_del_rows, values=scaled_top_rows)
    m6 = s.measure("H6_top10_del_weight_1.08")
    kept_h6, kept_m = decide(s, m6, kept_m, "H6")

    # Pass 8: H7 — Branch on H3: selective project_out on top-10 deleted rows if H3 was refused, else scale O_18.lin.weight by 1.05
    T = s.subject.tensors()
    if not kept_h3:
        h7 = s.hypothesis(
            "H7: Project out the CTC blank weight direction O_18.lin.weight[0] selectively from the top-10 deleted character rows",
            "CER decrease by 0.0010 to 0.0060",
            "project_out",
        )
        s.edit(
            "project_out",
            ["O_18.lin.weight"],
            hypothesis=h7,
            direction=T["O_18.lin.weight"][0].copy(),
            axis=0,
            index=top_del_rows,
        )
        m7 = s.measure("H7_project_out_blank_top10")
    else:
        h7 = s.hypothesis(
            "H7: Scale O_18.lin.weight by 1.05 to sharpen output logit contrast after full blank projection",
            "CER decrease by 0.0010 to 0.0060",
            "scale",
        )
        s.edit("scale", ["O_18.lin.weight"], hypothesis=h7, factor=1.05)
        m7 = s.measure("H7_scale_O18_weight_1.05")
    kept_h7, kept_m = decide(s, m7, kept_m, "H7")

    # Pass 9: H8 — Interpolate L_16 hidden biases (bias_hh_l0 and bias_hh_l0_reverse) with donor at alpha=0.10
    h8 = s.hypothesis(
        "H8: Interpolate L_16.layer.bias_hh_l0 and bias_hh_l0_reverse toward donor with alpha=0.10 (cosine ~0.89)",
        "CER decrease by 0.0010 to 0.0050",
        "interpolate",
    )
    l16_biases = ["L_16.layer.bias_hh_l0", "L_16.layer.bias_hh_l0_reverse"]
    s.edit(
        "interpolate",
        l16_biases,
        hypothesis=h8,
        alpha=0.10,
        per_tensor={n: {"other": donor_T[n]} for n in l16_biases},
    )
    m8 = s.measure("H8_L16_bias_hh_interp_0.10")
    kept_h8, kept_m = decide(s, m8, kept_m, "H8")

    # Pass 10: H9 — Align blank bias to donor delta (-0.353) if no blank bias edit was kept, else shrink shared-row bias residuals by 0.10
    T = s.subject.tensors()
    d9 = np.zeros(n_classes, dtype=np.float32)
    if not (kept_h1 or kept_h2):
        d9[0] = -0.353
        h9 = s.hypothesis(
            "H9: Shift O_18.lin.bias[0] by -0.353 to match the donor blank bias (+0.879)",
            "CER decrease by 0.0010 to 0.0050",
            "add",
        )
        label9 = "H9_donor_blank_bias_delta"
    else:
        shared_subj = np.array([r for r in sorted(row_map.keys()) if r != 0], dtype=np.int64)
        shared_don = np.array([row_map[int(r)] for r in shared_subj], dtype=np.int64)
        subj_b = T["O_18.lin.bias"][shared_subj]
        don_b = donor_T["O_18.lin.bias"][shared_don]
        subj_centered = subj_b - float(np.mean(subj_b))
        don_centered = don_b - float(np.mean(don_b))
        d9[shared_subj] = (0.10 * (don_centered - subj_centered)).astype(np.float32)
        h9 = s.hypothesis(
            "H9: Shrink mean-centered O_18.lin.bias residuals on the 172 shared non-blank character rows by 0.10 toward the donor profile",
            "CER decrease by 0.0010 to 0.0050",
            "add",
        )
        label9 = "H9_shared_bias_profile_0.10"
    s.edit("add", ["O_18.lin.bias"], hypothesis=h9, delta=d9)
    m9 = s.measure(label9)
    kept_h9, kept_m = decide(s, m9, kept_m, "H9")

    s.note(
        f"Finished 10 passes: baseline CER={m0.cer:.4f} (collapsed={m0.collapsed}) -> "
        f"final kept CER={kept_m.cer:.4f} (collapsed={kept_m.collapsed})"
    )
    s.freeze()


if __name__ == "__main__":
    main()
