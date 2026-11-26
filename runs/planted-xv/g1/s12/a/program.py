"""Weight-space repair program for pack s12, task (a) (budget 10)."""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    s = open_session(".")
    T = s.subject.tensors()
    cm = s.subject.codec_map()

    idx_e = cm["e"]
    idx_n = cm["n"]

    # Inspect alignment against parent donor for 'e' and 'n'
    parent = KrakenSubject(PARENT)
    pT = parent.tensors()
    pcm = parent.codec_map()
    p_idx_e = pcm["e"]
    p_idx_n = pcm["n"]

    w_sub_e = T["O_18.lin.weight"][idx_e].copy()
    w_sub_n = T["O_18.lin.weight"][idx_n].copy()
    b_sub_e = float(T["O_18.lin.bias"][idx_e])
    b_sub_n = float(T["O_18.lin.bias"][idx_n])

    w_par_e = pT["O_18.lin.weight"][p_idx_e].copy()
    w_par_n = pT["O_18.lin.weight"][p_idx_n].copy()
    b_par_e = float(pT["O_18.lin.bias"][p_idx_e])
    b_par_n = float(pT["O_18.lin.bias"][p_idx_n])

    cos_ee = cosine(w_sub_e, w_par_e)
    cos_nn = cosine(w_sub_n, w_par_n)
    cos_en = cosine(w_sub_e, w_par_n)
    cos_ne = cosine(w_sub_n, w_par_e)

    s.note(
        f"Pre-check O_18 rows 'e' (row {idx_e}) and 'n' (row {idx_n}) vs parent: "
        f"cos(sub_e, par_e)={cos_ee:.4f}, cos(sub_n, par_n)={cos_nn:.4f}, "
        f"cos(sub_e, par_n)={cos_en:.4f}, cos(sub_n, par_e)={cos_ne:.4f}; "
        f"bias sub=(e:{b_sub_e:.4f}, n:{b_sub_n:.4f}) vs par=(e:{b_par_e:.4f}, n:{b_par_n:.4f})"
    )

    # Pass 1: C0 baseline
    m0 = s.measure("C0")
    s.note(f"C0 baseline: CER={m0.cer:.4f}, WER={m0.wer:.4f}, collapsed={m0.collapsed}")

    # H1: Swap 'e' and 'n' rows in O_18.lin.weight and O_18.lin.bias
    h1 = s.hypothesis(
        "Output rows for 'e' and 'n' in O_18.lin.weight and O_18.lin.bias are swapped; "
        "swapping them back restores true class assignment.",
        "CER decreases by ~0.170-0.180 (from ~0.2443 to ~0.064-0.074)",
        "set_rows",
    )
    swapped_w = np.stack([w_sub_n, w_sub_e], axis=0).astype(T["O_18.lin.weight"].dtype)
    swapped_b = np.array([b_sub_n, b_sub_e], dtype=T["O_18.lin.bias"].dtype)
    s.edit(
        "set_rows",
        ["O_18.lin.weight", "O_18.lin.bias"],
        hypothesis=h1,
        index=[idx_e, idx_n],
        per_tensor={
            "O_18.lin.weight": {"values": swapped_w},
            "O_18.lin.bias": {"values": swapped_b},
        },
    )
    m1 = s.measure("H1_swap_e_n_weight_and_bias")
    if m1.cer < m0.cer and m1.collapsed <= m0.collapsed + 2:
        kept_h1 = s.keep(f"CER dropped from {m0.cer:.4f} to {m1.cer:.4f}")
    else:
        s.revert(f"H1 did not improve CER ({m0.cer:.4f} -> {m1.cer:.4f})")
        kept_h1 = False

    if kept_h1:
        # H2: Test whether O_18.lin.bias was unswapped in the planted defect
        # Only run if bias values actually differ
        if abs(b_sub_e - b_sub_n) > 1e-6:
            h2 = s.hypothesis(
                "Test whether the planted row swap affected only O_18.lin.weight by restoring "
                "the original O_18.lin.bias values for 'e' and 'n'.",
                "Small CER decrease (~0.0005-0.0020) if bias was not swapped; no improvement if bias was swapped too",
                "set_rows",
            )
            orig_b = np.array([b_sub_e, b_sub_n], dtype=T["O_18.lin.bias"].dtype)
            s.edit(
                "set_rows",
                ["O_18.lin.bias"],
                hypothesis=h2,
                index=[idx_e, idx_n],
                values=orig_b,
            )
            m2 = s.measure("H2_restore_orig_e_n_bias")
            if m2.cer < m1.cer and m2.collapsed <= m1.collapsed:
                s.keep(f"Restoring original bias further reduced CER ({m1.cer:.4f} -> {m2.cer:.4f})")
            else:
                s.revert(f"Bias was also swapped or difference not beneficial ({m1.cer:.4f} -> {m2.cer:.4f})")
    else:
        # Fallback H2: Transplant parent 'e' and 'n' rows into O_18
        h2 = s.hypothesis(
            "Fallback: transplant parent donor rows for 'e' and 'n' into O_18.lin.weight and O_18.lin.bias.",
            "CER decreases by ~0.140-0.170",
            "set_rows",
        )
        donor_w = np.stack([w_par_e, w_par_n], axis=0).astype(T["O_18.lin.weight"].dtype)
        donor_b = np.array([b_par_e, b_par_n], dtype=T["O_18.lin.bias"].dtype)
        s.edit(
            "set_rows",
            ["O_18.lin.weight", "O_18.lin.bias"],
            hypothesis=h2,
            index=[idx_e, idx_n],
            per_tensor={
                "O_18.lin.weight": {"values": donor_w},
                "O_18.lin.bias": {"values": donor_b},
            },
        )
        m2 = s.measure("H2_fallback_parent_e_n")
        if m2.cer < m0.cer and m2.collapsed <= m0.collapsed + 2:
            s.keep(f"Parent transplant reduced CER from {m0.cer:.4f} to {m2.cer:.4f}")
        else:
            s.revert(f"Parent transplant did not improve CER ({m0.cer:.4f} -> {m2.cer:.4f})")

    s.freeze()


if __name__ == "__main__":
    main()
