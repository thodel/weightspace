"""Weight-space session program for pack s14, task (a) (budget 10)."""
from __future__ import annotations

import numpy as np
from wsp.align import codec_row_map, common_names
from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def decide(s, m, kept_m, reason: str):
    """Apply our own checks (no rise in collapsed lines, CER strictly lower) and the harness keep rule."""
    if m.collapsed > kept_m.collapsed or m.cer >= kept_m.cer:
        s.revert(f"{reason}: reverted (CER {m.cer:.4f} vs {kept_m.cer:.4f}, collapsed {m.collapsed} vs {kept_m.collapsed})")
        return kept_m, False
    if s.keep(reason):
        return m, True
    return kept_m, False


def main() -> None:
    s = open_session(".")
    m0 = s.measure("C0")
    kept_m = m0

    parent = KrakenSubject(PARENT)
    T = s.subject.tensors()
    PT = parent.tensors()
    cm = s.subject.codec_map()
    pm = parent.codec_map()
    rmap = codec_row_map(cm, pm)
    rmap[0] = 0  # CTC blank row

    # --- Diagnostic scan against parent ---
    bad_out_rows: list[int] = []
    for rc, rp in sorted(rmap.items()):
        c = cosine(T["O_18.lin.weight"][rc], PT["O_18.lin.weight"][rp])
        nc = float(np.linalg.norm(T["O_18.lin.weight"][rc]))
        np_norm = float(np.linalg.norm(PT["O_18.lin.weight"][rp]))
        db = float(T["O_18.lin.bias"][rc] - PT["O_18.lin.bias"][rp])
        if c < 0.5 or nc < 0.25 * np_norm or nc > 4.0 * np_norm or abs(db) > 4.0:
            bad_out_rows.append(rc)

    bad_backbone: list[tuple[str, int, float]] = []
    for name in common_names(T, PT):
        if name.startswith("O_18"):
            continue
        w_c = T[name]
        w_p = PT[name]
        if w_c.ndim >= 2:
            for idx in range(w_c.shape[0]):
                c = cosine(w_c[idx], w_p[idx])
                nc = float(np.linalg.norm(w_c[idx]))
                np_n = float(np.linalg.norm(w_p[idx]))
                if c < 0.5 or (np_n > 1e-6 and (nc < 0.25 * np_n or nc > 3.5 * np_n)):
                    bad_backbone.append((name, idx, c))

    l16_diff = np.abs(T["L_16.layer.weight_ih_l0"] - PT["L_16.layer.weight_ih_l0"])
    l16_max_diff = float(np.max(l16_diff))
    l16_parent_max = float(np.max(np.abs(PT["L_16.layer.weight_ih_l0"])))
    s.note(
        f"Diagnostic scan vs parent: bad_out_rows={bad_out_rows}, "
        f"bad_backbone_count={len(bad_backbone)}, "
        f"L_16.weight_ih_l0 max_diff={l16_max_diff:.3f}, parent_abs_max={l16_parent_max:.3f}"
    )

    # --- H3a: If the scan detected any genuinely anomalous output rows or backbone rows, test repairing them first ---
    if bad_out_rows and s.budget_left > 0:
        T = s.subject.tensors()
        scales = [
            float(np.linalg.norm(T["O_18.lin.weight"][rc])) / max(float(np.linalg.norm(PT["O_18.lin.weight"][rp])), 1e-6)
            for rc, rp in rmap.items()
            if rc not in bad_out_rows
        ]
        med_scale = float(np.median(scales)) if scales else 1.0
        donor_w = np.stack([PT["O_18.lin.weight"][rmap[r]] * med_scale for r in bad_out_rows]).astype(np.float32)
        donor_b = np.array([PT["O_18.lin.bias"][rmap[r]] for r in bad_out_rows], dtype=np.float32)
        h = s.hypothesis(
            f"H3a: Restore anomalous output rows {bad_out_rows} from scaled parent rows",
            "CER decrease by 0.0010 to 0.0100",
            "set_rows",
        )
        s.edit(
            "set_rows",
            ["O_18.lin.weight", "O_18.lin.bias"],
            hypothesis=h,
            index=bad_out_rows,
            per_tensor={
                "O_18.lin.weight": {"values": donor_w},
                "O_18.lin.bias": {"values": donor_b},
            },
        )
        m = s.measure("H3a_out_rows")
        kept_m, _ = decide(s, m, kept_m, "H3a output row restoration")

    if bad_backbone and s.budget_left > 0:
        by_tensor: dict[str, list[int]] = {}
        for tname, idx, _ in bad_backbone:
            by_tensor.setdefault(tname, []).append(idx)
        names = sorted(by_tensor.keys())
        per_t = {
            n: {"index": by_tensor[n], "values": PT[n][np.asarray(by_tensor[n])].astype(np.float32)}
            for n in names
        }
        h = s.hypothesis(
            f"H3b: Restore anomalous backbone rows in {names} from parent",
            "CER decrease by 0.0010 to 0.0100",
            "set_rows",
        )
        s.edit("set_rows", names, hypothesis=h, per_tensor=per_t)
        m = s.measure("H3b_backbone_rows")
        kept_m, _ = decide(s, m, kept_m, "H3b backbone row restoration")

    # --- H1: CTC blank bias reduction (-0.15, and -0.30 if -0.15 helps or to test larger step) ---
    if s.budget_left > 0:
        T = s.subject.tensors()
        delta = np.zeros_like(T["O_18.lin.bias"])
        delta[0] = -0.15
        h1 = s.hypothesis(
            "H1a: Reduce CTC blank bias O_18.lin.bias[0] by 0.15 to counter deletion bias (length ratio 0.979)",
            "CER decrease by 0.0005 to 0.0020",
            "add",
        )
        s.edit("add", ["O_18.lin.bias"], hypothesis=h1, delta=delta)
        m1 = s.measure("H1a_blank_minus_015")
        kept_m, kept_h1a = decide(s, m1, kept_m, "H1a blank bias -0.15")

        if s.budget_left > 0 and (kept_h1a or m1.cer < m0.cer):
            T = s.subject.tensors()
            delta2 = np.zeros_like(T["O_18.lin.bias"])
            delta2[0] = -0.15 if kept_h1a else -0.25
            h1b = s.hypothesis(
                f"H1b: Further reduce CTC blank bias O_18.lin.bias[0] by {float(delta2[0]):.2f}",
                "CER decrease by 0.0005 to 0.0020",
                "add",
            )
            s.edit("add", ["O_18.lin.bias"], hypothesis=h1b, delta=delta2)
            m1b = s.measure("H1b_blank_step2")
            kept_m, _ = decide(s, m1b, kept_m, "H1b blank bias step 2")

    # --- H2: Space character bias boost and top-deleted character bias boost ---
    if " " in cm and s.budget_left > 0:
        T = s.subject.tensors()
        sp_idx = cm[" "]
        delta_sp = np.zeros_like(T["O_18.lin.bias"])
        delta_sp[sp_idx] = 0.20
        h2a = s.hypothesis(
            "H2a: Increase space character bias in O_18.lin.bias by +0.20 (78 deletions vs 25 insertions at C0)",
            "CER decrease by 0.0005 to 0.0020",
            "add",
        )
        s.edit("add", ["O_18.lin.bias"], hypothesis=h2a, delta=delta_sp)
        m2a = s.measure("H2a_space_plus_020")
        kept_m, _ = decide(s, m2a, kept_m, "H2a space bias +0.20")

    if s.budget_left > 0:
        T = s.subject.tensors()
        delta_top = np.zeros_like(T["O_18.lin.bias"])
        for ch in (" ", "e", "n", "i", "r", "s"):
            if ch in cm:
                delta_top[cm[ch]] = 0.12
        h2b = s.hypothesis(
            "H2b: Boost biases of top-6 deleted characters (' ', 'e', 'n', 'i', 'r', 's') by +0.12",
            "CER decrease by 0.0005 to 0.0015",
            "add",
        )
        s.edit("add", ["O_18.lin.bias"], hypothesis=h2b, delta=delta_top)
        m2b = s.measure("H2b_top_deleted_plus_012")
        kept_m, _ = decide(s, m2b, kept_m, "H2b top-6 deleted chars bias +0.12")

    # --- H3c: Test transplanting C_3.co.bias from parent (cosine 0.679, rel L2 1.087) ---
    if s.budget_left > 0:
        h3c = s.hypothesis(
            "H3c: Transplant C_3.co.bias from parent (lowest cosine 0.679 and highest rel L2 1.087 in diff table)",
            "CER change by -0.0010 to 0.0000",
            "transplant",
        )
        s.edit("transplant", ["C_3.co.bias"], hypothesis=h3c, donor=PT["C_3.co.bias"])
        m3c = s.measure("H3c_C3_bias_parent")
        kept_m, _ = decide(s, m3c, kept_m, "H3c transplant C_3.co.bias")

    # --- H3d: If L_16.layer.weight_ih_l0 has outlier differences vs parent (> 1.5), test interpolating it slightly ---
    if s.budget_left > 0 and l16_max_diff > 1.5:
        h3d = s.hypothesis(
            f"H3d: Interpolate L_16.layer.weight_ih_l0 10% toward parent (subject abs max 4.702, max diff {l16_max_diff:.2f})",
            "CER change by -0.0010 to 0.0000",
            "interpolate",
        )
        s.edit(
            "interpolate",
            ["L_16.layer.weight_ih_l0"],
            hypothesis=h3d,
            other=PT["L_16.layer.weight_ih_l0"],
            alpha=0.10,
        )
        m3d = s.measure("H3d_L16_ih_interp_010")
        kept_m, _ = decide(s, m3d, kept_m, "H3d interpolate L_16.layer.weight_ih_l0 alpha=0.10")

    # --- H4: Mild weight-space interpolation of shared backbone tensors toward parent (alpha=0.05) ---
    if s.budget_left > 0:
        shared = [n for n in common_names(s.subject.tensors(), PT) if not n.startswith("O_18") and "bias_ih" not in n]
        per_t = {n: {"other": PT[n]} for n in shared}
        h4 = s.hypothesis(
            "H4: Interpolate shared conv/BiLSTM backbone tensors toward parent with alpha=0.05",
            "CER change by -0.0010 to +0.0025",
            "interpolate",
        )
        s.edit("interpolate", shared, hypothesis=h4, alpha=0.05, per_tensor=per_t)
        m4 = s.measure("H4_backbone_interp_005")
        kept_m, _ = decide(s, m4, kept_m, "H4 backbone interpolation alpha=0.05")

    s.note(f"Finished session: baseline CER={m0.cer:.4f}, final kept CER={kept_m.cer:.4f}, passes used={10 - s.budget_left}")
    s.freeze()


if __name__ == "__main__":
    main()
