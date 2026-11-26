"""Weight-space repair program for pack s13, task (a) (budget 10)."""
from __future__ import annotations

import numpy as np
from wsp.kraken_subject import KrakenSubject
from wsp.run import open_session

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"


def _confusion_count(confusions, ref_ch: str, hyp_ch: str) -> int:
    """Safely extract count for (ref_ch, hyp_ch) from Metrics.confusions."""
    if not confusions:
        return 0
    ref_aliases = {ref_ch, "∅"} if ref_ch == "" else {ref_ch}
    hyp_aliases = {hyp_ch, "∅"} if hyp_ch == "" else {hyp_ch}
    if isinstance(confusions, dict):
        for (r, h), c in confusions.items():
            if r in ref_aliases and h in hyp_aliases:
                return int(c)
        return 0
    for item in confusions:
        if isinstance(item, dict):
            r = item.get("ref", item.get("reference"))
            h = item.get("hyp", item.get("hypothesis"))
            if r in ref_aliases and h in hyp_aliases:
                return int(item.get("count", 0))
        elif isinstance(item, (tuple, list)):
            if len(item) == 3 and item[0] in ref_aliases and item[1] in hyp_aliases:
                return int(item[2])
            if len(item) == 2 and isinstance(item[0], (tuple, list)) and len(item[0]) == 2:
                r, h = item[0]
                if r in ref_aliases and h in hyp_aliases:
                    return int(item[1])
    return 0


def main() -> None:
    s = open_session(".")
    cm = s.subject.codec_map()
    row_n = cm["n"]

    # Pass 1: C0 baseline measurement
    m0 = s.measure("C0")
    ins_n_0 = _confusion_count(m0.confusions, "", "n")
    del_n_0 = _confusion_count(m0.confusions, "n", "")
    s.note(
        f"C0 baseline: CER={m0.cer:.4f}, WER={m0.wer:.4f}, collapsed={m0.collapsed}, "
        f"ins_n={ins_n_0}, del_n={del_n_0}"
    )

    # H1: Sign-flip repair of O_18.lin.weight row for 'n' (cos = -0.991 vs parent, rel L2 = 2.042)
    T = s.subject.tensors()
    w_n_orig = T["O_18.lin.weight"][[row_n]].copy()
    h1 = s.hypothesis(
        "Row 'n' of O_18.lin.weight has its sign inverted (cos -0.991 vs parent, norm 8.33 vs 7.96), "
        "causing 'n' to fire on non-'n' frames (7,667 false 'n' insertions); negating row cm['n'] restores it.",
        "CER decreases by ~0.65 to ~0.70 (from 0.7456 to ~0.05-0.09) and collapsed lines drop from 79 to ~0.",
        "set_rows",
    )
    s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h1, index=[row_n], values=-w_n_orig)
    m1 = s.measure("H1_negate_row_n_weight")
    ins_n_1 = _confusion_count(m1.confusions, "", "n")
    del_n_1 = _confusion_count(m1.confusions, "n", "")

    if m1.cer < m0.cer and m1.collapsed <= m0.collapsed:
        kept1 = s.keep(
            f"H1 reduced CER from {m0.cer:.4f} to {m1.cer:.4f}, "
            f"collapsed {m0.collapsed} -> {m1.collapsed}, ins_n {ins_n_0} -> {ins_n_1}"
        )
    else:
        s.revert(f"H1 did not improve CER ({m0.cer:.4f} -> {m1.cer:.4f})")
        kept1 = False

    best_cer = m1.cer if kept1 else m0.cer

    # Branch: only test H2 / H3 if H1 failed or if 'n' insertions/deletions or CER remain anomalously high
    if not kept1 or best_cer > 0.12 or ins_n_1 > 100 or del_n_1 > 100:
        parent = KrakenSubject(PARENT)
        pm = parent.codec_map()
        pT = parent.tensors()
        p_row_n = pm["n"]

        # H2: Test restoring parent bias for 'n' in O_18.lin.bias if 'n' rate is still skewed
        b_parent_n = np.asarray([pT["O_18.lin.bias"][p_row_n]], dtype=np.float32)
        h2 = s.hypothesis(
            "If 'n' insertions/deletions or CER remain elevated after H1, O_18.lin.bias[cm['n']] may also be corrupted; "
            "transplant parent bias for 'n'.",
            "CER decreases by 0.002 to 0.020.",
            "set_rows",
        )
        s.edit("set_rows", ["O_18.lin.bias"], hypothesis=h2, index=[row_n], values=b_parent_n)
        m2 = s.measure("H2_parent_bias_n")
        if m2.cer < best_cer - 0.0005:
            if s.keep(f"H2 reduced CER from {best_cer:.4f} to {m2.cer:.4f}"):
                best_cer = m2.cer
        else:
            s.revert(f"H2 bias edit did not meaningfully improve CER ({best_cer:.4f} -> {m2.cer:.4f})")

        # H3: If CER is still > 0.12, transplant parent's O_18.lin.weight row for 'n' (scaled to child norm)
        if best_cer > 0.12:
            T = s.subject.tensors()
            child_norm = float(np.linalg.norm(T["O_18.lin.weight"][row_n]))
            p_vec = pT["O_18.lin.weight"][p_row_n].astype(np.float64)
            p_norm = float(np.linalg.norm(p_vec))
            donor_row = (p_vec * (child_norm / p_norm if p_norm > 0 else 1.0)).astype(np.float32)[None, :]

            h3 = s.hypothesis(
                "Child row 'n' in O_18.lin.weight remains defective; transplant parent row 'n' rescaled to child norm.",
                "CER decreases by 0.60 to 0.69 vs baseline.",
                "set_rows",
            )
            s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h3, index=[row_n], values=donor_row)
            m3 = s.measure("H3_donor_row_n_weight")
            if m3.cer < best_cer - 0.0005:
                if s.keep(f"H3 donor row 'n' reduced CER from {best_cer:.4f} to {m3.cer:.4f}"):
                    best_cer = m3.cer
            else:
                s.revert(f"H3 donor row 'n' did not improve CER ({best_cer:.4f} -> {m3.cer:.4f})")
    else:
        s.note(
            f"H1 completely resolved the 'n' sign-flip defect (CER={m1.cer:.4f}, collapsed={m1.collapsed}, "
            f"ins_n={ins_n_1}, del_n={del_n_1}); skipping H2/H3 to preserve exact child weights."
        )

    s.freeze()


if __name__ == "__main__":
    main()
