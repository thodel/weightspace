"""Round 0 program (program_01.py):
Pass 1: C0 baseline verification.
Pass 2: H1 — Restore CTC blank bias O_18.lin.bias[0] to parent (hertwig-v2) value (+0.8792).
Pass 3: H2 — Character-aligned output layer (O_18.lin.weight, O_18.lin.bias) midpoint merge of clean parent and clean sibling.
Pass 4: H3 — Full-network character-aligned midpoint merge (0.5 * parent + 0.5 * sibling) across all 34 tensors.
"""
from pathlib import Path
import json
import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows
from wsp.metrics import score, page_of, bootstrap_delta

ROOT = Path("/home/tobias/Repo/weightspace")
PARENT_PATH = ROOT / "models/kraken/kraken-bern-hertwig-v2.mlmodel"
SIBLING_PATH = ROOT / "models/kraken/kraken-bern-schlussredaktion-v1.mlmodel"


def report_pass(s, m, label, kept_collapsed):
    log_entry = s._log[-1]
    vs_k = log_entry.get("vs_kept") or {}
    vs_b = log_entry.get("vs_baseline") or {}

    # Read the latest pass readings to compute per-hand CER and page-clustered CI
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

    print(f"=== Pass {s._used}: {label} ===")
    print(f"  Overall CER: {m.cer:.4f} ({m.char_edits}/{m.ref_chars}) | WER: {m.wer:.4f} | collapsed: {m.collapsed} (kept_collapsed={kept_collapsed}) | len_ratio: {m.length_ratio:.4f}")
    print(f"  Hertwig CER: {m_h.cer:.4f} ({m_h.char_edits}/{m_h.ref_chars}, collapsed={m_h.collapsed}) | Schlussred. CER: {m_s.cer:.4f} ({m_s.char_edits}/{m_s.ref_chars}, collapsed={m_s.collapsed})")
    if vs_k:
        print(f"  vs_kept:     delta={vs_k['delta_cer']:+.4f} | line_ci95=[{vs_k['ci95'][0]:+.4f}, {vs_k['ci95'][1]:+.4f}] | page_ci95=[{bd_cl['ci95_cluster'][0]:+.4f}, {bd_cl['ci95_cluster'][1]:+.4f}]")
    if vs_b:
        print(f"  vs_baseline: delta={vs_b['delta_cer']:+.4f} | line_ci95=[{vs_b['ci95'][0]:+.4f}, {vs_b['ci95'][1]:+.4f}]")
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
    orig_T = {k: v.copy() for k, v in s.subject.tensors().items()}
    cms = s.subject.codec_map()

    parent = KrakenSubject(PARENT_PATH)
    sibling = KrakenSubject(SIBLING_PATH)
    Tp = parent.tensors()
    Tb = sibling.tensors()
    cmp = parent.codec_map()
    cmb = sibling.codec_map()

    # Align output rows by character (plus row 0 = CTC blank)
    rm_p = codec_row_map(cms, cmp)
    rm_p[0] = 0
    rm_b = codec_row_map(cms, cmb)
    rm_b[0] = 0

    n_rows = orig_T["O_18.lin.weight"].shape[0]
    Wp_al = gather_rows(Tp["O_18.lin.weight"], rm_p, n_rows, fill=orig_T["O_18.lin.weight"])
    bp_al = gather_rows(Tp["O_18.lin.bias"], rm_p, n_rows, fill=orig_T["O_18.lin.bias"])
    Wb_al = gather_rows(Tb["O_18.lin.weight"], rm_b, n_rows, fill=orig_T["O_18.lin.weight"])
    bb_al = gather_rows(Tb["O_18.lin.bias"], rm_b, n_rows, fill=orig_T["O_18.lin.bias"])

    # For the 2 chars in sibling but not parent ('½', '̊'), use sibling's rows in parent-aligned arrays when merging
    for ch in ("½", "̊"):
        if ch in cms and ch in cmb:
            idx_s = cms[ch]
            Wp_al[idx_s] = Wb_al[idx_s]
            bp_al[idx_s] = bb_al[idx_s]

    # Pass 1: C0 baseline
    m0 = s.measure("C0")
    kept_collapsed = m0.collapsed
    report_pass(s, m0, "C0 (baseline)", kept_collapsed)
    if round(m0.cer, 4) != 0.1456:
        s.note(f"C0 CER {m0.cer:.4f} did not reproduce expected 0.1456; stopping.")
        print(f"ERROR: C0 CER {m0.cer:.4f} != 0.1456. Stopping.")
        return

    # Pass 2: H1 — Blank bias restoration to parent value (+0.8792)
    h1 = s.hypothesis(
        "H1: Restore CTC blank bias O_18.lin.bias[0] from +0.9976 to parent hertwig-v2 value +0.8792 (delta -0.1184). "
        "Parent hertwig-v2 was trained on the clean 22.10 Hertwig training pages (103 practice lines); subject combined-v1 "
        "inflated blank bias by +0.118 due to label noise on 44/50 mismatched Hertwig pages.",
        "Practice CER decreases by -0.0020 to -0.0060 across both hands (fewer deletions) with no rise in collapsed lines.",
        "set_rows"
    )
    s.edit("set_rows", ["O_18.lin.bias"], hypothesis=h1, index=[0], values=np.array([Tp["O_18.lin.bias"][0]], dtype=np.float32))
    m1 = s.measure("H1_blank_bias_parent")
    report_pass(s, m1, "H1_blank_bias_parent", kept_collapsed)
    _, kept_collapsed = decide_edit(s, m1, kept_collapsed, "H1 blank bias restored to parent lowered CER")

    # Pass 3: H2 — Full output-layer character-aligned midpoint of parent and sibling
    h2 = s.hypothesis(
        "H2: Transplant character-aligned midpoint (0.5 * parent + 0.5 * sibling) into O_18.lin.weight and O_18.lin.bias "
        "(keeping subject's row for 'ƒ' and sibling's rows for '½' and '̊'). Parent hertwig-v2 saw the 103 Hertwig practice lines "
        "cleanly; sibling schlussredaktion-v1 saw the 197 Schlussredaktion practice lines cleanly. Undoes label-noise inflation "
        "of rare/umlaut classifier rows and restores clean output weights and biases.",
        "Practice CER decreases by -0.0030 to -0.0080 (primarily on Hertwig, -0.0050 to -0.0150; Schlussredaktion 0.0000 to -0.0030).",
        "transplant"
    )
    W_o18_mid = (0.5 * Wp_al + 0.5 * Wb_al).astype(np.float32)
    b_o18_mid = (0.5 * bp_al + 0.5 * bb_al).astype(np.float32)
    s.edit(
        "transplant",
        ["O_18.lin.weight", "O_18.lin.bias"],
        hypothesis=h2,
        per_tensor={
            "O_18.lin.weight": {"donor": W_o18_mid},
            "O_18.lin.bias": {"donor": b_o18_mid},
        }
    )
    m2 = s.measure("H2_O18_parent_sibling_midpoint")
    report_pass(s, m2, "H2_O18_parent_sibling_midpoint", kept_collapsed)
    _, kept_collapsed = decide_edit(s, m2, kept_collapsed, "H2 O_18 clean donor midpoint lowered CER")

    # Pass 4: H3 — Full-network character-aligned midpoint merge (0.5 * parent + 0.5 * sibling) across all 34 tensors
    h3 = s.hypothesis(
        "H3: Full-network character-aligned task-vector midpoint merge (0.5 * parent + 0.5 * sibling) across all 34 tensors "
        "(C_0..C_9, L_12..L_16, O_18). Parent hertwig-v2 was trained on clean Hertwig training pages (103 practice lines) and "
        "sibling schlussredaktion-v1 was fine-tuned from parent on clean Schlussredaktion training pages (197 practice lines). "
        "Because subject combined-v1 corrupted its conv and BiLSTM stack (especially L_12) on 44/50 mismatched Hertwig image-text pages, "
        "replacing all tensors with the clean 50/50 parent-sibling interpolation removes the feature-level corruption while combining both hands.",
        "Practice CER drops by -0.0150 to -0.0300 vs baseline (Hertwig CER dropping from ~0.205 to ~0.130-0.155, Schlussredaktion CER ~0.120-0.135, collapsed dropping from 29 to 21-25).",
        "transplant"
    )
    all_names = list(orig_T.keys())
    per_tensor_mid = {}
    for name in all_names:
        if name == "O_18.lin.weight":
            per_tensor_mid[name] = {"donor": W_o18_mid}
        elif name == "O_18.lin.bias":
            per_tensor_mid[name] = {"donor": b_o18_mid}
        else:
            per_tensor_mid[name] = {"donor": (0.5 * Tp[name] + 0.5 * Tb[name]).astype(np.float32)}

    s.edit("transplant", all_names, hypothesis=h3, per_tensor=per_tensor_mid)
    m3 = s.measure("H3_full_parent_sibling_midpoint_a050")
    report_pass(s, m3, "H3_full_parent_sibling_midpoint_a050", kept_collapsed)
    _, kept_collapsed = decide_edit(s, m3, kept_collapsed, "H3 full-network 50/50 clean donor merge lowered CER")

    print(f"End of program_01.py: budget_used={s._used}/{s.budget}, budget_left={s.budget_left}")


if __name__ == "__main__":
    main()
