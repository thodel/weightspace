"""Round 3 program (program_04.py — final program):
Pass 9:  H8 — Uniform three-way blend (0.35 * subject + 0.25 * parent + 0.40 * sibling) across all feature and
              classifier weights (C_0..L_16 and character-aligned O_18.lin.weight), paired with clean donor
              O_18.lin.bias (0.38 * parent + 0.62 * sibling) and CTC blank bias O_18.lin.bias[0] = +0.840.
Pass 10: H9 — Frobenius/row norm restoration + CTC blank bias calibration (-0.10) on top of the kept three-way blend
              to counter the ~1% norm shrinkage of 3-way weight averaging and recover borderline character deletions.
Ends with s.freeze().
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

    # Pass 9: H8 — Uniform three-way blend (0.35 * subject + 0.25 * parent + 0.40 * sibling) across all
    # feature and classifier weights (including O_18.lin.weight), with clean O_18.lin.bias (0.38 * par + 0.62 * sib)
    # and blank bias O_18.lin.bias[0] = +0.840.
    h8 = s.hypothesis(
        "H8: Uniform three-way blend (0.35 * subject + 0.25 * parent + 0.40 * sibling) across all 33 weight and "
        "recurrent bias tensors (C_0..L_16 and character-aligned O_18.lin.weight), paired with clean donor bias "
        "(0.38 * parent + 0.62 * sibling) in O_18.lin.bias and CTC blank bias O_18.lin.bias[0] = +0.840. "
        "Parent hertwig-v2 saw the 103 Hertwig practice lines cleanly; sibling schlussredaktion-v1 saw the 197 "
        "Schlussredaktion practice lines cleanly. Keeping uniform weights across C_0..O_18 avoids cross-layer mismatch "
        "while reducing the label-noise residual from 50% to 35% and aligning O_18.lin.weight with L_16's 35% subject component.",
        "Practice CER decreases by -0.0010 to -0.0035 vs Pass 5 kept state (1468 edits, CER 0.1356), improving both "
        "Hertwig and Schlussredaktion with collapsed <= 21.",
        "transplant"
    )
    per_tensor_h8 = {}
    for name in all_names:
        if name == "O_18.lin.weight":
            per_tensor_h8[name] = {"donor": (0.35 * Ts[name] + 0.25 * Wp_al + 0.40 * Wb_al).astype(np.float32)}
        elif name == "O_18.lin.bias":
            b_clean = (0.38 * bp_al + 0.62 * bb_al).astype(np.float32)
            b_clean[0] = np.float32(0.840)
            per_tensor_h8[name] = {"donor": b_clean}
        else:
            per_tensor_h8[name] = {"donor": (0.35 * Ts[name] + 0.25 * Tp[name] + 0.40 * Tb[name]).astype(np.float32)}

    s.edit("transplant", all_names, hypothesis=h8, per_tensor=per_tensor_h8)
    m8 = s.measure("H8_uniform_3way_35sub_25par_40sib")
    report_pass(s, m8, "H8_uniform_3way_35sub_25par_40sib", kept_collapsed)
    kept_h8, kept_collapsed = decide_edit(s, m8, kept_collapsed, "H8 uniform 35/25/40 three-way blend lowered CER")

    # Pass 10: H9 — Restore Frobenius norms of conv/LSTM weight tensors and row norms of O_18.lin.weight
    # to the weighted average of the constituent model norms, plus trim blank bias O_18.lin.bias[0] by -0.10.
    h9 = s.hypothesis(
        "H9: Restore Frobenius norms of all conv/BiLSTM weight tensors (C_0..L_16) and per-row L2 norms of O_18.lin.weight "
        "on the current kept three-way blend (built from subject combined-v1, parent hertwig-v2 trained on 103 Hertwig "
        "practice lines, and sibling schlussredaktion-v1 trained on 197 Schlussredaktion practice lines) to match the "
        "convex combination of the constituent tensor/row norms (countering the ~0.6%-1.1% norm shrinkage of averaging "
        "non-collinear weights), and lower CTC blank bias O_18.lin.bias[0] by -0.10. This restores logit dynamic range "
        "and reduces remaining deletion errors.",
        "Practice CER decreases by -0.0010 to -0.0025 vs the kept state (fewer deletions) with collapsed <= kept_collapsed.",
        "transplant"
    )
    cur_T = {k: v.copy() for k, v in s.subject.tensors().items()}
    if kept_h8:
        ws, wp, wb = 0.35, 0.25, 0.40
        wo_s, wo_p, wo_b = 0.35, 0.25, 0.40
    else:
        ws, wp, wb = 0.50, 0.20, 0.30
        wo_s, wo_p, wo_b = 0.00, 0.40, 0.60

    per_tensor_h9 = {}
    edit_names_h9 = []
    for name in all_names:
        if "bias_ih" in name:
            continue
        if name == "O_18.lin.weight":
            W_cur = cur_T[name].copy()
            for r in range(n_rows):
                n_cur = float(np.linalg.norm(W_cur[r]))
                n_target = (
                    wo_s * float(np.linalg.norm(Ts[name][r]))
                    + wo_p * float(np.linalg.norm(Wp_al[r]))
                    + wo_b * float(np.linalg.norm(Wb_al[r]))
                )
                if n_cur > 1e-8:
                    W_cur[r] = (W_cur[r] * (n_target / n_cur)).astype(np.float32)
            per_tensor_h9[name] = {"donor": W_cur}
            edit_names_h9.append(name)
        elif name == "O_18.lin.bias":
            b_cur = cur_T[name].copy()
            b_cur[0] = np.float32(float(b_cur[0]) - 0.10)
            per_tensor_h9[name] = {"donor": b_cur}
            edit_names_h9.append(name)
        elif "weight" in name:
            W_cur = cur_T[name].copy()
            n_cur = float(np.linalg.norm(W_cur))
            n_target = (
                ws * float(np.linalg.norm(Ts[name]))
                + wp * float(np.linalg.norm(Tp[name]))
                + wb * float(np.linalg.norm(Tb[name]))
            )
            if n_cur > 1e-8:
                W_cur = (W_cur * (n_target / n_cur)).astype(np.float32)
            per_tensor_h9[name] = {"donor": W_cur}
            edit_names_h9.append(name)

    s.edit("transplant", edit_names_h9, hypothesis=h9, per_tensor=per_tensor_h9)
    m9 = s.measure("H9_norm_restored_blank_minus_010")
    report_pass(s, m9, "H9_norm_restored_blank_minus_010", kept_collapsed)
    _, kept_collapsed = decide_edit(s, m9, kept_collapsed, "H9 norm restoration + blank -0.10 lowered CER")

    summary = s.freeze()
    print("=== Session Frozen ===")
    print(json.dumps({
        "name": summary["name"],
        "used": summary["used"],
        "budget": summary["budget"],
        "edits": summary["edits"],
        "kept": summary["kept"],
        "start_digest": summary["start_digest"],
        "final_digest": summary["final_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
