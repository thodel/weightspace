"""Weight-space session program for pack s04, task (a) (budget 10)."""
from __future__ import annotations

import numpy as np
from wsp.align import codec_row_map
from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"


def decide_edit(s, m, best_m, reason: str, min_cer_drop: float = 0.0005):
    """Apply our own checks on m.cer and m.collapsed, then delegate to s.keep()."""
    if m.collapsed > best_m.collapsed:
        s.revert(f"reverted ({reason}): collapsed lines increased {best_m.collapsed} -> {m.collapsed}")
        return False, best_m
    cer_drop = best_m.cer - m.cer
    if cer_drop < min_cer_drop and m.collapsed == best_m.collapsed:
        s.revert(f"reverted ({reason}): CER change {-cer_drop:+.5f} smaller than threshold -{min_cer_drop:.4f}")
        return False, best_m
    kept = s.keep(f"{reason}: CER {best_m.cer:.4f} -> {m.cer:.4f}, collapsed {best_m.collapsed} -> {m.collapsed}")
    if kept:
        return True, m
    return False, best_m


def main() -> None:
    s = open_session(".")
    parent = KrakenSubject(PARENT)

    # Pass 1: Baseline measurement (C0)
    m0 = s.measure("C0")
    best_m = m0
    s.note(f"C0 baseline: CER={m0.cer:.4f}, WER={m0.wer:.4f}, collapsed={m0.collapsed}, len_ratio={m0.length_ratio:.4f}")

    T = s.subject.tensors()
    P = parent.tensors()
    cm = s.subject.codec_map()
    pm = parent.codec_map()
    rmap = codec_row_map(cm, pm)

    # --- Diagnostic scan of subject vs parent ---
    # 1. Check L_16.layer.weight_ih_l0 abs max against parent
    l16_w = T["L_16.layer.weight_ih_l0"]
    l16_p = P["L_16.layer.weight_ih_l0"]
    l16_diff = l16_w - l16_p
    max_idx = np.unravel_index(int(np.argmax(np.abs(l16_w))), l16_w.shape)
    max_diff_idx = np.unravel_index(int(np.argmax(np.abs(l16_diff))), l16_diff.shape)

    # 2. Check row-wise cosines across all shared conv/LSTM weights
    worst_row_info = ("", -1, 1.0, 0.0)  # (name, row, cos, rel_l2)
    for name in sorted(T.keys()):
        if name.startswith("O_18") or name not in P or T[name].shape != P[name].shape:
            continue
        if T[name].ndim >= 2:
            w_flat = T[name].reshape(T[name].shape[0], -1).astype(np.float64)
            p_flat = P[name].reshape(P[name].shape[0], -1).astype(np.float64)
            for r in range(w_flat.shape[0]):
                nw = np.linalg.norm(w_flat[r])
                np_ = np.linalg.norm(p_flat[r])
                if nw > 1e-6 and np_ > 1e-6:
                    cos = float(np.dot(w_flat[r], p_flat[r]) / (nw * np_))
                    rel = float(np.linalg.norm(w_flat[r] - p_flat[r]) / np_)
                    if cos < worst_row_info[2]:
                        worst_row_info = (name, r, cos, rel)

    # 3. Check output bias deltas for shared characters (especially space and top deleted chars)
    ob_t = T["O_18.lin.bias"]
    ob_p = P["O_18.lin.bias"]
    shared_b_deltas = {ch: float(ob_t[ia] - ob_p[ib]) for ch, ia in cm.items() if ia in rmap for ib in [rmap[ia]]}
    med_b_delta = float(np.median(list(shared_b_deltas.values()))) if shared_b_deltas else 0.0
    space_idx = cm.get(" ")
    space_b_delta = shared_b_deltas.get(" ", 0.0)

    # Find any shared character whose bias dropped anomalously relative to parent
    anomalous_bias_chars = [
        (ch, cm[ch], d)
        for ch, d in shared_b_deltas.items()
        if abs(d - med_b_delta) > 3.0
    ]

    s.note(
        f"Diagnostics: L_16.weight_ih_l0 abs_max={float(np.abs(l16_w[max_idx])):.3f} at {max_idx} "
        f"(parent={float(l16_p[max_idx]):.3f}); max_diff={float(l16_diff[max_diff_idx]):.3f} at {max_diff_idx}. "
        f"Worst shared tensor row cosine: {worst_row_info[0]}[{worst_row_info[1]}] cos={worst_row_info[2]:.4f} "
        f"rel_L2={worst_row_info[3]:.4f}. "
        f"O_18 bias median delta={med_b_delta:+.3f}, space bias delta={space_b_delta:+.3f}, "
        f"anomalous_bias_chars={len(anomalous_bias_chars)}."
    )

    # --- H1: Localized outlier repair if a genuine parent-child spike or corrupted row is detected ---
    if worst_row_info[2] < 0.70 and s.budget_left > 0:
        tname, r_idx, r_cos, r_rel = worst_row_info
        h1 = s.hypothesis(
            f"H1: Row {r_idx} of {tname} has anomalous cosine {r_cos:.4f} (rel L2 {r_rel:.4f}) vs parent; "
            "transplanting that row from parent repairs localized corruption.",
            "decrease CER by 0.0020 to 0.0150",
            "set_rows",
        )
        s.edit("set_rows", [tname], hypothesis=h1, index=[r_idx], values=P[tname][[r_idx]])
        m1 = s.measure("H1_row_repair")
        _, best_m = decide_edit(s, m1, best_m, f"H1 repair {tname}[{r_idx}]")
    elif float(np.abs(l16_diff[max_diff_idx])) > 1.5 and s.budget_left > 0:
        r_idx = int(max_diff_idx[0])
        h1 = s.hypothesis(
            f"H1: L_16.layer.weight_ih_l0 row {r_idx} contains a large parent-child element jump "
            f"({float(l16_diff[max_diff_idx]):+.3f}); restoring row {r_idx} from parent tests if abs max 4.232 is a defect.",
            "decrease CER by 0.0020 to 0.0100",
            "set_rows",
        )
        s.edit(
            "set_rows",
            ["L_16.layer.weight_ih_l0"],
            hypothesis=h1,
            index=[r_idx],
            values=P["L_16.layer.weight_ih_l0"][[r_idx]],
        )
        m1 = s.measure("H1_l16_outlier_row")
        _, best_m = decide_edit(s, m1, best_m, f"H1 restore L_16.weight_ih_l0[{r_idx}]")

    # --- H2: CTC blank bias reduction on O_18.lin.bias[0] ---
    if s.budget_left > 0:
        T = s.subject.tensors()
        delta_b = np.zeros_like(T["O_18.lin.bias"])
        delta_b[0] = -0.20
        h2 = s.hypothesis(
            "H2: Lower CTC blank bias O_18.lin.bias[0] by -0.20 (from +1.232 toward +1.032) to test whether "
            "reducing blank dominance recovers borderline collapsed lines and reduces deletions.",
            "decrease CER by 0.0005 to 0.0030",
            "add",
        )
        s.edit("add", ["O_18.lin.bias"], hypothesis=h2, delta=delta_b)
        m2 = s.measure("H2_blank_bias_m020")
        kept_h2, best_m = decide_edit(s, m2, best_m, "H2 blank bias -0.20")

        # Branch: if -0.20 helped, try an additional -0.15; if -0.20 was too strong, try -0.113 (exact parent delta)
        if kept_h2 and s.budget_left > 0:
            T = s.subject.tensors()
            delta_b2 = np.zeros_like(T["O_18.lin.bias"])
            delta_b2[0] = -0.15
            h2b = s.hypothesis(
                "H2b: Lower CTC blank bias by a further -0.15 after initial improvement.",
                "decrease CER by 0.0005 to 0.0020",
                "add",
            )
            s.edit("add", ["O_18.lin.bias"], hypothesis=h2b, delta=delta_b2)
            m2b = s.measure("H2b_blank_bias_m035")
            _, best_m = decide_edit(s, m2b, best_m, "H2b blank bias additional -0.15")

    # --- H3: Space character bias boost on O_18.lin.bias[cm[' ']] ---
    if space_idx is not None and s.budget_left > 0:
        T = s.subject.tensors()
        delta_sp = np.zeros_like(T["O_18.lin.bias"])
        # If space bias dropped notably vs parent, restore that drop; otherwise test +0.25
        step = float(-space_b_delta) if space_b_delta < -0.35 else 0.25
        delta_sp[space_idx] = step
        h3 = s.hypothesis(
            f"H3: Increase space character output bias O_18.lin.bias[{space_idx}] by {step:+.3f} "
            "to reduce top confusion ' ' -> empty (124 deletions vs 19 insertions).",
            "decrease CER by 0.0005 to 0.0025",
            "add",
        )
        s.edit("add", ["O_18.lin.bias"], hypothesis=h3, delta=delta_sp)
        m3 = s.measure("H3_space_bias")
        _, best_m = decide_edit(s, m3, best_m, f"H3 space bias {step:+.3f}")

    # --- H4: Gentle parent interpolation on L_16 weights (alpha=0.05) ---
    if s.budget_left > 0:
        l16_names = [
            "L_16.layer.weight_ih_l0",
            "L_16.layer.weight_hh_l0",
            "L_16.layer.bias_hh_l0",
            "L_16.layer.weight_ih_l0_reverse",
            "L_16.layer.weight_hh_l0_reverse",
            "L_16.layer.bias_hh_l0_reverse",
        ]
        per_tensor = {n: {"other": P[n]} for n in l16_names}
        h4 = s.hypothesis(
            "H4: Interpolate L_16 BiLSTM parameters slightly toward parent donor (alpha=0.05) "
            "to regularize high-magnitude recurrent weights.",
            "decrease CER by 0.0005 to 0.0020",
            "interpolate",
        )
        s.edit("interpolate", l16_names, hypothesis=h4, per_tensor=per_tensor, alpha=0.05)
        m4 = s.measure("H4_L16_interp_005")
        _, best_m = decide_edit(s, m4, best_m, "H4 L_16 parent interpolation alpha=0.05")

    s.note(f"Final kept state: CER={best_m.cer:.4f}, WER={best_m.wer:.4f}, collapsed={best_m.collapsed}")
    s.freeze()


if __name__ == "__main__":
    main()
