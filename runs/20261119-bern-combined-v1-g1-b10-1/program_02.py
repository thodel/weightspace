"""Round 1 program (program_02.py):
Pass 5: H4 — Three-way blend of kept subject body (50%) and clean donor merge (20% parent + 30% sibling)
             in C_0..L_16, with character-aligned O_18 set to 0.40 * parent + 0.60 * sibling.
Pass 6: H5 — Pure clean donor interpolation (0.32 * parent + 0.68 * sibling, alpha=0.68) across all 34 tensors.
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
    # Find the metrics of the last kept pass from log.jsonl
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

    # Pass 5: H4 — Three-way blend: 0.50 * subject + 0.20 * parent + 0.30 * sibling for C_0..L_16,
    # and 0.40 * parent + 0.60 * sibling for O_18.
    h4 = s.hypothesis(
        "H4: Three-way blend of subject combined-v1 (0.50), clean parent hertwig-v2 (0.20), and clean sibling "
        "schlussredaktion-v1 (0.30) across C_0..L_16, paired with character-aligned 0.40 * parent + 0.60 * sibling "
        "in O_18. Parent saw the 103 Hertwig practice lines cleanly; sibling saw the 197 Schlussredaktion practice lines "
        "cleanly; subject saw clean Schlussredaktion plus 44/50 mismatched Hertwig pages. Moving halfway from the Pass 3 "
        "kept state toward the 40/60 clean donor merge cuts the orthogonal label-noise residual in C_0..L_16 by 50% while "
        "preserving the shared Schlussredaktion feature basin.",
        "Practice CER decreases by -0.0080 to -0.0180 vs Pass 3 kept state (Hertwig CER dropping from 0.1968 to ~0.155-0.175, "
        "Schlussredaktion CER staying near ~0.115-0.122, collapsed <= 27).",
        "transplant"
    )
    per_tensor_h4 = {}
    for name in all_names:
        if name == "O_18.lin.weight":
            per_tensor_h4[name] = {"donor": (0.40 * Wp_al + 0.60 * Wb_al).astype(np.float32)}
        elif name == "O_18.lin.bias":
            per_tensor_h4[name] = {"donor": (0.40 * bp_al + 0.60 * bb_al).astype(np.float32)}
        else:
            per_tensor_h4[name] = {"donor": (0.50 * Ts[name] + 0.20 * Tp[name] + 0.30 * Tb[name]).astype(np.float32)}

    s.edit("transplant", all_names, hypothesis=h4, per_tensor=per_tensor_h4)
    m4 = s.measure("H4_blend_50sub_20par_30sib")
    report_pass(s, m4, "H4_blend_50sub_20par_30sib", kept_collapsed)
    _, kept_collapsed = decide_edit(s, m4, kept_collapsed, "H4 three-way blend (50% sub, 20% par, 30% sib) lowered CER")

    # Pass 6: H5 — Pure clean donor merge at alpha=0.68 (0.32 * parent + 0.68 * sibling) across all 34 tensors
    h5 = s.hypothesis(
        "H5: Pure clean donor interpolation 0.32 * parent + 0.68 * sibling (alpha=0.68) across all 34 tensors "
        "(character-aligned in O_18). Parent was trained on clean Hertwig (103 practice lines) and sibling on clean "
        "Schlussredaktion (197 practice lines, 68.6% of practice characters). Quadratic interpolation of the per-hand "
        "error curve through alpha=0 (parent), alpha=0.50 (Pass 4), and alpha=1.0 (sibling) predicts the total CER "
        "minimum along the parent-sibling segment lies near alpha=0.68.",
        "Practice CER reaches ~0.1350-0.1380 (-0.0050 to -0.0090 vs baseline, and below Pass 4's 0.1461), with "
        "Hertwig CER ~0.165-0.175, Schlussredaktion CER ~0.118-0.123, and collapsed ~23-25.",
        "transplant"
    )
    per_tensor_h5 = {}
    for name in all_names:
        if name == "O_18.lin.weight":
            per_tensor_h5[name] = {"donor": (0.32 * Wp_al + 0.68 * Wb_al).astype(np.float32)}
        elif name == "O_18.lin.bias":
            per_tensor_h5[name] = {"donor": (0.32 * bp_al + 0.68 * bb_al).astype(np.float32)}
        else:
            per_tensor_h5[name] = {"donor": (0.32 * Tp[name] + 0.68 * Tb[name]).astype(np.float32)}

    s.edit("transplant", all_names, hypothesis=h5, per_tensor=per_tensor_h5)
    m5 = s.measure("H5_clean_donor_merge_a068")
    report_pass(s, m5, "H5_clean_donor_merge_a068", kept_collapsed)
    _, kept_collapsed = decide_edit(s, m5, kept_collapsed, "H5 clean donor merge alpha=0.68 lowered CER")

    print(f"End of program_02.py: budget_used={s._used}/{s.budget}, budget_left={s.budget_left}")


if __name__ == "__main__":
    main()
