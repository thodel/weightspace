"""Round 2 program (program_03.py):
Pass 7: H6 — Layer-graded three-way blend:
             C_0..C_9: 0.40 * subject + 0.25 * parent + 0.35 * sibling
             L_12:     0.30 * subject + 0.30 * parent + 0.40 * sibling (stronger denoising where orthogonal residual is largest)
             L_14:     0.50 * subject + 0.18 * parent + 0.32 * sibling
             L_16:     0.55 * subject + 0.15 * parent + 0.30 * sibling
             O_18:     0.38 * parent + 0.62 * sibling (character-aligned)
Pass 8: H7 — Trim CTC blank bias O_18.lin.bias[0] by -0.050 on top of the current kept state to reduce remaining deletions (804 del vs 177 ins).
"""
from pathlib import Path
import json
import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows
from wsp.metrics import score, page_of, bootstrap_delta

ROOT = Path("/home/tobias/Repo/weightspace")
SUBJECT_PATH = ROOT / "models/kraken/kraken-bern-combined-v1.mlmodel"
PARENT_PATH = ROOT / "models/kraken/kraken-bern-hertwig-v2.mlmodel"
SIBLING_PATH = ROOT / "models/kraken/kraken-bern-schlussredaktion-v1.mlmodel"


def get_kept_collapsed(s):
    full = [json.loads(l) for l in (s.run_dir / "log.jsonl").read_text().splitlines() if l.strip()]
    kept_col = 29
    last_pass_col = 29
    for e in full:
        if e["kind"] == "pass":
            last_pass_col = e["metrics"]["collapsed"]
            if e.get("label") == "C0":
                kept_col = last_pass_col
        elif e["kind"] == "decision" and e.get("kept"):
            kept_col = last_pass_col
    return kept_col


def report_pass(s, m, label, kept_collapsed):
    log_entry = s._log[-1]
    vs_k = log_entry.get("vs_kept") or {}
    vs_b = log_entry.get("vs_baseline") or {}

    pass_file = s.run_dir / "readings" / f"pass_{s._used:03d}.jsonl"
    preds = [json.loads(line) for line in pass_file.read_text().splitlines() if line.strip()]
    hyps = [p["text"] for p in preds]
    refs = [it["text"] for it in s.practice]
    keys = [it["key"] for it in s.practice]

    h_idx = [i for i, k in enumerate(keys) if "Hertwig" in k]
    s_idx = [i for i, k in enumerate(keys) if "Schlussredaktion" in k]

    m_h = score([hyps[i] for i in h_idx], [refs[i] for i in h_idx], blocks=False)
    m_s = score([hyps[i] for i in s_idx], [refs[i] for i in s_idx], blocks=False)

    clusters = [page_of(k) for k in keys]
    bd_cl = bootstrap_delta(s._kept_lines, s._last_lines, resamples=2000, seed=s.seed, clusters=clusters)
    bd_base_cl = bootstrap_delta(s._baseline_lines, s._last_lines, resamples=2000, seed=s.seed, clusters=clusters)

    print(f"=== Pass {s._used}: {label} ===")
    print(f"  Overall CER: {m.cer:.4f} ({m.char_edits}/{m.ref_chars}) | WER: {m.wer:.4f} | collapsed: {m.collapsed} (kept_collapsed={kept_collapsed}) | len_ratio: {m.length_ratio:.4f}")
    print(f"  Hertwig CER: {m_h.cer:.4f} ({m_h.char_edits}/{m_h.ref_chars}, collapsed={m_h.collapsed}) | Schlussred. CER: {m_s.cer:.4f} ({m_s.char_edits}/{m_s.ref_chars}, collapsed={m_s.collapsed})")
    if vs_k:
        print(f"  vs_kept:     delta={vs_k['delta_cer']:+.4f} | line_ci95=[{vs_k['ci95'][0]:+.4f}, {vs_k['ci95'][1]:+.4f}] | page_ci95=[{bd_cl['ci95_cluster'][0]:+.4f}, {bd_cl['ci95_cluster'][1]:+.4f}]")
    if vs_b:
        print(f"  vs_baseline: delta={vs_b['delta_cer']:+.4f} | line_ci95=[{vs_b['ci95'][0]:+.4f}, {vs_b['ci95'][1]:+.4f}] | page_ci95=[{bd_base_cl['ci95_cluster'][0]:+.4f}, {bd_base_cl['ci95_cluster'][1]:+.4f}]")
    print(f"  Top confusions: {m.confusions[:8]}")


def decide_edit(s, m, kept_collapsed, reason_keep):
    if m.collapsed > kept_collapsed:
        s.revert(f"collapsed count rose ({kept_collapsed} -> {m.collapsed})")
        print(f"  -> REVERTED (collapsed rose {kept_collapsed} -> {m.collapsed})")
        return False, kept_collapsed
    kept = s.keep(reason_keep)
    if kept:
        print(f"  -> KEPT ({reason_keep}); new kept_collapsed={m.collapsed}")
        return True, m.collapsed
    else:
        why = s._log[-1].get("reason", "rule refused")
        print(f"  -> REFUSED by keep rule ({why})")
        return False, kept_collapsed


def main():
    s = open_session(".")
    kept_collapsed = get_kept_collapsed(s)
    print(f"Reopened session: budget_used={s._used}/{s.budget}, kept_collapsed={kept_collapsed}")

    orig_subj = KrakenSubject(SUBJECT_PATH)
    parent = KrakenSubject(PARENT_PATH)
    sibling = KrakenSubject(SIBLING_PATH)

    Ts = orig_subj.tensors()
    Tp = parent.tensors()
    Tb = sibling.tensors()

    cms = orig_subj.codec_map()
    cmp = parent.codec_map()
    cmb = sibling.codec_map()

    rm_p = codec_row_map(cms, cmp)
    rm_p[0] = 0
    rm_b = codec_row_map(cms, cmb)
    rm_b[0] = 0

    n_rows = Ts["O_18.lin.weight"].shape[0]
    Wp_al = gather_rows(Tp["O_18.lin.weight"], rm_p, n_rows, fill=Ts["O_18.lin.weight"])
    bp_al = gather_rows(Tp["O_18.lin.bias"], rm_p, n_rows, fill=Ts["O_18.lin.bias"])
    Wb_al = gather_rows(Tb["O_18.lin.weight"], rm_b, n_rows, fill=Ts["O_18.lin.weight"])
    bb_al = gather_rows(Tb["O_18.lin.bias"], rm_b, n_rows, fill=Ts["O_18.lin.bias"])

    for ch in ("½", "̊"):
        if ch in cms and ch in cmb:
            idx_s = cms[ch]
            Wp_al[idx_s] = Wb_al[idx_s]
            bp_al[idx_s] = bb_al[idx_s]

    all_names = list(Ts.keys())

    # Pass 7: H6 — Layer-graded three-way blend
    h6 = s.hypothesis(
        "H6: Layer-graded three-way blend of subject combined-v1, clean parent hertwig-v2 (trained on 103 Hertwig practice lines), "
        "and clean sibling schlussredaktion-v1 (trained on 197 Schlussredaktion practice lines). Because L_12 has the lowest "
        "cosine (0.252) and largest orthogonal label-noise residual (||r||/||p||=0.227) from the 44/50 mismatched Hertwig pages, "
        "while L_14 and L_16 carry cleaner shared representations (cos 0.475 and 0.577), we weight (sub, par, sib) as: "
        "C_0..C_9=(0.40, 0.25, 0.35), L_12=(0.30, 0.30, 0.40), L_14=(0.50, 0.18, 0.32), L_16=(0.55, 0.15, 0.30), "
        "and character-aligned O_18=(0.00, 0.38, 0.62).",
        "Practice CER decreases by -0.0010 to -0.0040 vs Pass 5 kept state (1468 edits, CER 0.1356), improving both Hertwig "
        "(via stronger L_12 denoising) and Schlussredaktion (via higher L_14/L_16/O_18 task projection), with collapsed <= 21.",
        "transplant"
    )
    block_weights = {
        "C_0":  (0.40, 0.25, 0.35),
        "C_3":  (0.40, 0.25, 0.35),
        "C_6":  (0.40, 0.25, 0.35),
        "C_9":  (0.40, 0.25, 0.35),
        "L_12": (0.30, 0.30, 0.40),
        "L_14": (0.50, 0.18, 0.32),
        "L_16": (0.55, 0.15, 0.30),
    }
    per_tensor_h6 = {}
    for name in all_names:
        if name == "O_18.lin.weight":
            per_tensor_h6[name] = {"donor": (0.38 * Wp_al + 0.62 * Wb_al).astype(np.float32)}
        elif name == "O_18.lin.bias":
            per_tensor_h6[name] = {"donor": (0.38 * bp_al + 0.62 * bb_al).astype(np.float32)}
        else:
            blk = name.split(".")[0]
            ws, wp, wb = block_weights[blk]
            per_tensor_h6[name] = {"donor": (ws * Ts[name] + wp * Tp[name] + wb * Tb[name]).astype(np.float32)}

    s.edit("transplant", all_names, hypothesis=h6, per_tensor=per_tensor_h6)
    m6 = s.measure("H6_layer_graded_3way_blend")
    report_pass(s, m6, "H6_layer_graded_3way_blend", kept_collapsed)
    _, kept_collapsed = decide_edit(s, m6, kept_collapsed, "H6 layer-graded three-way blend lowered CER")

    # Pass 8: H7 — Trim CTC blank bias O_18.lin.bias[0] by -0.050 on top of the current kept state
    cur_blank_bias = float(s.subject.tensors()["O_18.lin.bias"][0])
    new_blank_bias = cur_blank_bias - 0.050
    h7 = s.hypothesis(
        f"H7: Trim CTC blank bias O_18.lin.bias[0] by -0.050 (from {cur_blank_bias:.4f} to {new_blank_bias:.4f}) on top of "
        "the kept three-way blend (built from subject, parent hertwig-v2 trained on 103 Hertwig practice lines, and sibling "
        "schlussredaktion-v1 trained on 197 Schlussredaktion practice lines). Even after the three-way merge, deletions "
        "outnumber insertions 4.5 to 1 (804 del vs 177 ins; length ratio 0.939). Lowering the blank bias slightly should "
        "recover borderline character and space emissions.",
        "Practice CER decreases by -0.0005 to -0.0020 vs the kept three-way blend (fewer deletions on both hands) with collapsed <= 21.",
        "set_rows"
    )
    s.edit("set_rows", ["O_18.lin.bias"], hypothesis=h7, index=[0], values=np.array([new_blank_bias], dtype=np.float32))
    m7 = s.measure("H7_blank_bias_minus_005")
    report_pass(s, m7, "H7_blank_bias_minus_005", kept_collapsed)
    _, kept_collapsed = decide_edit(s, m7, kept_collapsed, "H7 blank bias -0.050 trim lowered CER")

    print(f"End of program_03.py: budget_used={s._used}/{s.budget}, budget_left={s.budget_left}")


if __name__ == "__main__":
    main()
