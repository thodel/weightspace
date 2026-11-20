"""Weight-space repair program for pack s06, task (a) (budget 10).

Diagnosis from pack readings:
- describe.md: `O_18.lin.weight` row for `'e'` has norm 0.00 (completely zeroed out).
- diff_parent_to_subject.md: `'e'` row in `O_18.lin.weight` has rel L2 1.000, cos 0.000,
  norm 7.64 -> 0.00, bias delta -0.142.
- c0_practice.md: CER 0.1623 (2,088 edits on 12,867 chars), dominated by `'e' -> ∅` (910)
  and `'e' -> {a, i, o, r, c, u}` (131+90+52+36+35+32 = 376), totaling ~1,286 edits.
"""
from __future__ import annotations

import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"


def count_confusion(confusions: list[dict], ref: str, hyp: str) -> int:
    for item in confusions:
        if item.get("ref") == ref and item.get("hyp") == hyp:
            return int(item.get("count", 0))
    return 0


def main() -> None:
    s = open_session(".")
    parent = KrakenSubject(PARENT)

    cm = s.subject.codec_map()
    pm = parent.codec_map()
    p_tensors = parent.tensors()

    # Pass 1: Baseline C0
    m0 = s.measure("C0")
    s.note(f"Baseline C0: CER={m0.cer:.4f}, WER={m0.wer:.4f}, collapsed={m0.collapsed}")

    e_child = cm["e"]
    e_parent = pm["e"]

    p_e_w = p_tensors["O_18.lin.weight"][e_parent : e_parent + 1].copy()
    p_e_b = p_tensors["O_18.lin.bias"][e_parent : e_parent + 1].copy()
    c_e_b = s.subject.tensors()["O_18.lin.bias"][e_child : e_child + 1].copy()

    # Pass 2 (H1): Restore 'e' row in O_18.lin.weight only (keeping child's fine-tuned O_18.lin.bias['e'])
    h1 = s.hypothesis(
        "H1: The output weight row for 'e' in O_18.lin.weight was zeroed out (norm 0.00 vs parent 7.64). "
        "Transplanting the parent 'e' weight row into O_18.lin.weight repairs 'e' recognition.",
        "CER will decrease by 0.090 to 0.105 (from ~0.1623 to ~0.060-0.072)",
        "set_rows",
    )
    s.edit(
        "set_rows",
        ["O_18.lin.weight"],
        hypothesis=h1,
        index=[e_child],
        values=p_e_w,
    )
    m1 = s.measure("H1_restore_e_weight")
    kept1 = s.keep("Restore zeroed 'e' row in O_18.lin.weight from parent donor")
    s.note(f"H1 result: CER={m1.cer:.4f}, WER={m1.wer:.4f}, collapsed={m1.collapsed}, kept={kept1}")

    # Pass 3 (H2): Also test parent's bias for 'e' (delta = +0.142 above child bias)
    h2 = s.hypothesis(
        "H2: Transplanting the parent 'e' bias in O_18.lin.bias alongside the parent 'e' weight row "
        "restores the exact parent logit offset (+0.142) and reduces residual 'e -> ∅' deletions.",
        "CER will decrease by 0.0005 to 0.0030",
        "set_rows",
    )
    if kept1:
        s.edit(
            "set_rows",
            ["O_18.lin.bias"],
            hypothesis=h2,
            index=[e_child],
            values=p_e_b,
        )
    else:
        # Fallback if weight-only failed: set both weight and bias together
        s.edit(
            "set_rows",
            ["O_18.lin.weight", "O_18.lin.bias"],
            hypothesis=h2,
            index=[e_child],
            per_tensor={
                "O_18.lin.weight": {"values": p_e_w},
                "O_18.lin.bias": {"values": p_e_b},
            },
        )
    m2 = s.measure("H2_restore_e_bias")
    kept2 = s.keep("Restore parent 'e' bias in O_18.lin.bias")
    s.note(f"H2 result: CER={m2.cer:.4f}, kept={kept2}")

    m_cur = m2 if kept2 else (m1 if kept1 else m0)

    # Pass 4 & 5 (H3): Calibrate restored 'e' weight scale and bias to child L_16 features
    # Check residual 'e' deletions vs insertions in m_cur.confusions
    e_del = count_confusion(m_cur.confusions, "e", "")
    e_ins = count_confusion(m_cur.confusions, "", "e")
    s.note(f"After 'e' restoration: residual e->∅ = {e_del}, ∅->e = {e_ins}")

    # Scale the 'e' weight row slightly up (1.08x) to compensate for L_16 drift (cos ~0.95)
    h3a = s.hypothesis(
        "H3a: Because L_16 drifted from parent to child (cosine ~0.95-0.98), the inner product with "
        "the parent 'e' weight row is slightly attenuated. Scaling the 'e' weight row by 1.08 "
        "restores logit sharpness for 'e'.",
        "CER will decrease by 0.0003 to 0.0020",
        "set_rows",
    )
    s.edit(
        "set_rows",
        ["O_18.lin.weight"],
        hypothesis=h3a,
        index=[e_child],
        values=(p_e_w * 1.08).astype(np.float32),
    )
    m3a = s.measure("H3a_scale_e_1.08")
    kept3a = s.keep("Scale 'e' row in O_18.lin.weight by 1.08")
    if kept3a:
        m_cur = m3a
    else:
        # Try 0.93x scale if 1.08x was not kept
        h3b = s.hypothesis(
            "H3b: Scaling the parent 'e' weight row by 0.93 reduces over-triggering of 'e' on "
            "similar Kurrent vowels/strokes.",
            "CER will decrease by 0.0003 to 0.0020",
            "set_rows",
        )
        s.edit(
            "set_rows",
            ["O_18.lin.weight"],
            hypothesis=h3b,
            index=[e_child],
            values=(p_e_w * 0.93).astype(np.float32),
        )
        m3b = s.measure("H3b_scale_e_0.93")
        kept3b = s.keep("Scale 'e' row in O_18.lin.weight by 0.93")
        if kept3b:
            m_cur = m3b

    # Check if a targeted bias nudge on 'e' helps based on residual confusions
    e_del = count_confusion(m_cur.confusions, "e", "")
    e_ins = count_confusion(m_cur.confusions, "", "e")
    cur_e_b = s.subject.tensors()["O_18.lin.bias"][e_child : e_child + 1].copy()
    nudge = 0.25 if e_del >= e_ins else -0.25

    h3c = s.hypothesis(
        f"H3c: Adjusting 'e' bias in O_18.lin.bias by {nudge:+.2f} balances residual "
        f"'e' deletions ({e_del}) vs insertions ({e_ins}).",
        "CER will decrease by 0.0003 to 0.0015",
        "set_rows",
    )
    s.edit(
        "set_rows",
        ["O_18.lin.bias"],
        hypothesis=h3c,
        index=[e_child],
        values=(cur_e_b + nudge).astype(np.float32),
    )
    m3c = s.measure("H3c_nudge_e_bias")
    kept3c = s.keep(f"Nudge 'e' bias by {nudge:+.2f}")
    if kept3c:
        m_cur = m3c

    # Pass 7 (H4): Lower CTC blank bias (row 0) slightly to reduce overall deletion bias
    if s.budget_left >= 2:
        cur_blank_b = s.subject.tensors()["O_18.lin.bias"][0:1].copy()
        h4 = s.hypothesis(
            "H4: Lowering the CTC blank bias (row 0 of O_18.lin.bias) by -0.20 reduces character "
            "deletions (' ' -> ∅, 'n' -> ∅, 'r' -> ∅).",
            "CER will decrease by 0.0003 to 0.0020",
            "set_rows",
        )
        s.edit(
            "set_rows",
            ["O_18.lin.bias"],
            hypothesis=h4,
            index=[0],
            values=(cur_blank_b - 0.20).astype(np.float32),
        )
        m4 = s.measure("H4_lower_blank_bias")
        kept4 = s.keep("Lower CTC blank bias by 0.20")
        if kept4:
            m_cur = m4

    # Pass 8 (H5): Boost space (' ') bias if space deletion remains prominent
    if s.budget_left >= 2 and " " in cm:
        sp_idx = cm[" "]
        sp_del = count_confusion(m_cur.confusions, " ", "")
        sp_ins = count_confusion(m_cur.confusions, "", " ")
        if sp_del > sp_ins:
            cur_sp_b = s.subject.tensors()["O_18.lin.bias"][sp_idx : sp_idx + 1].copy()
            h5 = s.hypothesis(
                f"H5: Space deletions (' ' -> ∅: {sp_del}) exceed insertions ({sp_ins}). "
                "Increasing O_18.lin.bias for ' ' by +0.30 reduces missed word boundaries.",
                "CER will decrease by 0.0003 to 0.0015",
                "set_rows",
            )
            s.edit(
                "set_rows",
                ["O_18.lin.bias"],
                hypothesis=h5,
                index=[sp_idx],
                values=(cur_sp_b + 0.30).astype(np.float32),
            )
            m5 = s.measure("H5_boost_space_bias")
            kept5 = s.keep("Increase space bias by +0.30")
            if kept5:
                m_cur = m5

    # Pass 9 (H6): Test C_3.co.bias transplant (had rel L2 1.0871 in diff_parent_to_subject.md)
    if s.budget_left >= 1:
        h6 = s.hypothesis(
            "H6: C_3.co.bias had the highest relative L2 change among conv tensors (1.0871, cos 0.679). "
            "Test transplanting parent C_3.co.bias to verify whether it was a defect or benign drift.",
            "CER will decrease by 0.0000 to 0.0010",
            "transplant",
        )
        s.edit(
            "transplant",
            ["C_3.co.bias"],
            hypothesis=h6,
            donor=p_tensors["C_3.co.bias"].copy(),
        )
        m6 = s.measure("H6_transplant_C3_bias")
        kept6 = s.keep("Transplant parent C_3.co.bias")
        if kept6:
            m_cur = m6

    s.note(f"Final state: CER={m_cur.cer:.4f}, WER={m_cur.wer:.4f}, collapsed={m_cur.collapsed}, passes_left={s.budget_left}")
    s.freeze()


if __name__ == "__main__":
    main()
