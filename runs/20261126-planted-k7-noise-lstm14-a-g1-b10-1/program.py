"""Weight-space session program for pack s02, task (a) (budget 10)."""
from __future__ import annotations

import numpy as np
from wsp.align import codec_row_map
from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"


def decide(s, m, best_cer: float, best_collapsed: int, label: str) -> tuple[bool, float, int]:
    """Check CER and collapsed lines before delegating to the paired 95% CI keep rule."""
    if m.collapsed > best_collapsed:
        s.revert(f"{label}: collapsed lines rose ({best_collapsed} -> {m.collapsed})")
        return False, best_cer, best_collapsed
    if m.cer >= best_cer:
        s.revert(f"{label}: CER did not improve ({best_cer:.4f} -> {m.cer:.4f})")
        return False, best_cer, best_collapsed
    kept = s.keep(f"{label}: CER improved {best_cer:.4f} -> {m.cer:.4f} without extra collapsed lines")
    if kept:
        return True, m.cer, m.collapsed
    return False, best_cer, best_collapsed


def main() -> None:
    s = open_session(".")
    parent = KrakenSubject(PARENT)
    T = s.subject.tensors()
    P = parent.tensors()
    cm = s.subject.codec_map()
    pm = parent.codec_map()
    rmap = codec_row_map(cm, pm)

    # Pass 1: C0 baseline
    m0 = s.measure("C0")
    best_cer = m0.cer
    best_collapsed = m0.collapsed
    s.note(
        f"C0 baseline: CER={m0.cer:.4f}, WER={m0.wer:.4f}, collapsed={m0.collapsed}. "
        "Tensor inspection shows high parent-child cosine similarity across all blocks "
        "(0.919-0.9995), indicating no catastrophic planted defect; testing H1-H6 for "
        "output deletion calibration and parent-alignment regularization."
    )

    # H1: Revert the +0.113 blank bias shift relative to parent
    h1 = s.hypothesis(
        "H1: Reduce O_18.lin.bias[0] (<blank>) by 0.113 to revert its positive drift from the parent and reduce character deletions",
        "CER decrease by 0.0005 to 0.0020 (delta CER in [-0.0020, -0.0005])",
        "add",
    )
    delta_h1 = np.zeros_like(T["O_18.lin.bias"])
    delta_h1[0] = -0.113
    s.edit("add", ["O_18.lin.bias"], hypothesis=h1, delta=delta_h1)
    m1 = s.measure("H1_blank_bias")
    kept1, best_cer, best_collapsed = decide(s, m1, best_cer, best_collapsed, "H1")

    # H2: Boost space bias by +0.15 to counter the dominant ' ' -> empty deletion confusion
    if " " in cm:
        h2 = s.hypothesis(
            "H2: Add +0.15 to O_18.lin.bias for the space character (' ') to reduce missed word spaces (' ' -> empty: 103 vs empty -> ' ': 22)",
            "CER decrease by 0.0005 to 0.0020 (delta CER in [-0.0020, -0.0005])",
            "add",
        )
        delta_h2 = np.zeros_like(T["O_18.lin.bias"])
        delta_h2[cm[" "]] = 0.15
        s.edit("add", ["O_18.lin.bias"], hypothesis=h2, delta=delta_h2)
        m2 = s.measure("H2_space_bias")
        kept2, best_cer, best_collapsed = decide(s, m2, best_cer, best_collapsed, "H2")

    # H3: Scale weakest high-frequency lowercase character rows ('e', 'n', 's', 'i', 'r', 't', 'o') by 1.05
    freq_chars = [c for c in ("e", "n", "s", "i", "r", "t", "o") if c in cm]
    if freq_chars:
        h3 = s.hypothesis(
            "H3: Scale O_18.lin.weight rows for high-deletion weak-norm lowercase letters ('e','n','s','i','r','t','o') by 1.05",
            "CER decrease by 0.0005 to 0.0020 (delta CER in [-0.0020, -0.0005])",
            "set_rows",
        )
        idx_h3 = [cm[c] for c in freq_chars]
        vals_h3 = (T["O_18.lin.weight"][idx_h3] * 1.05).astype(np.float32)
        s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h3, index=idx_h3, values=vals_h3)
        m3 = s.measure("H3_scale_weak_rows")
        kept3, best_cer, best_collapsed = decide(s, m3, best_cer, best_collapsed, "H3")

    # H4: Transplant C_3.co.bias from parent (rel L2 = 1.0871, cos = 0.67929)
    h4 = s.hypothesis(
        "H4: Transplant C_3.co.bias from parent (the only tensor with rel L2 > 1.0 against parent)",
        "CER decrease by 0.0005 to 0.0030 (delta CER in [-0.0030, -0.0005])",
        "transplant",
    )
    s.edit("transplant", ["C_3.co.bias"], hypothesis=h4, donor=P["C_3.co.bias"])
    m4 = s.measure("H4_c3_bias_transplant")
    kept4, best_cer, best_collapsed = decide(s, m4, best_cer, best_collapsed, "H4")

    # H5: Interpolate shared O_18 rows 10% toward parent
    h5 = s.hypothesis(
        "H5: Interpolate the 225 shared rows of O_18.lin.weight and O_18.lin.bias 10% toward parent (alpha=0.10)",
        "CER decrease by 0.0005 to 0.0020 (delta CER in [-0.0020, -0.0005])",
        "set_rows",
    )
    shared_child = [0] + sorted(rmap.keys())
    shared_parent = [0] + [rmap[i] for i in sorted(rmap.keys())]
    w_vals = (0.90 * T["O_18.lin.weight"][shared_child] + 0.10 * P["O_18.lin.weight"][shared_parent]).astype(np.float32)
    b_vals = (0.90 * T["O_18.lin.bias"][shared_child] + 0.10 * P["O_18.lin.bias"][shared_parent]).astype(np.float32)
    s.edit(
        "set_rows",
        ["O_18.lin.weight", "O_18.lin.bias"],
        hypothesis=h5,
        index=shared_child,
        per_tensor={
            "O_18.lin.weight": {"values": w_vals},
            "O_18.lin.bias": {"values": b_vals},
        },
    )
    m5 = s.measure("H5_o18_shared_interp")
    kept5, best_cer, best_collapsed = decide(s, m5, best_cer, best_collapsed, "H5")

    # H6: Interpolate L_14 weights 5% toward parent
    h6 = s.hypothesis(
        "H6: Interpolate L_14 weight tensors 5% toward parent (alpha=0.05) to test if L_14's 0.41 rel L2 drift is over-adaptation",
        "CER decrease by 0.0005 to 0.0020 (delta CER in [-0.0020, -0.0005])",
        "interpolate",
    )
    l14_names = [
        "L_14.layer.weight_ih_l0",
        "L_14.layer.weight_hh_l0",
        "L_14.layer.weight_ih_l0_reverse",
        "L_14.layer.weight_hh_l0_reverse",
    ]
    s.edit(
        "interpolate",
        l14_names,
        hypothesis=h6,
        alpha=0.05,
        per_tensor={n: {"other": P[n]} for n in l14_names},
    )
    m6 = s.measure("H6_l14_interp")
    kept6, best_cer, best_collapsed = decide(s, m6, best_cer, best_collapsed, "H6")

    s.note(f"Finished session: best CER={best_cer:.4f} (start {m0.cer:.4f}), collapsed={best_collapsed}.")
    s.freeze()


if __name__ == "__main__":
    main()
