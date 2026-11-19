# Notes — 20261119-bern-combined-v1-g1-b10-1

## 2026-11-19 08:04:00 UTC — Round 0 (Pre-registration & Initial Analysis)

- **Model ID:** `gemini-4-argon`
- **Settings:** Provider default reasoning settings (`provider: "gemini"`, `model: "gemini-4-argon"`, `options: {}`), called via `wsp.programmer`, sandboxed CPU execution (`CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `unshare -Un`).
- **Files read:**
  - `runs/20261119-bern-combined-v1-g1-b10-1/PROGRAMMER_BRIEF.md`
  - `runs/20261119-bern-combined-v1-g1-b10-1/preregistration.md`, `notes.md`, `program.py`, `programmer.json`, `session.json`, `env.txt`
  - `runs/20261119-bern-combined-v1-g1-b10-1/pack/c0_c1.md`, `readings_budget0.md`, `per_line_edits.json`
  - `runs/20261119-bern-combined-v1-g1-b10-1/pack/describe_bern-combined-v1.md`, `describe_bern-hertwig-v2.md`, `describe_bern-schlussredaktion-v1.md`
  - `runs/20261119-bern-combined-v1-g1-b10-1/pack/diff_bern-hertwig-v2__bern-combined-v1.md`, `diff_bern-hertwig-v2__bern-schlussredaktion-v1.md`
  - `data/practice/bern-train-300.jsonl`, `data/practice/bern-train-300.meta.json`
  - `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`, `wsp/subject.py`, `wsp/align.py`, `wsp/metrics.py`, `wsp/data.py`
  - Read-only tensor & codec inspection of `models/kraken/kraken-bern-combined-v1.mlmodel`, `models/kraken/kraken-bern-hertwig-v2.mlmodel`, and `models/kraken/kraken-bern-schlussredaktion-v1.mlmodel`.

### Key Findings from Budget-0 Readings & Weight Geometry
1. **The defect in `kraken-bern-combined-v1`:**
   - `combined-v1` was fine-tuned from `hertwig-v2` on both hands, but 44 of its 50 Hertwig training pages came from the corrupted 25.10 export where images were paired with another page's transcription.
   - Consequently, on the 103 Hertwig practice lines (with correct 22.10 images), `combined-v1` degrades to CER `0.2050` compared to `0.1137` for its parent `hertwig-v2`. On the 197 Schlussredaktion practice lines, `combined-v1` achieves `0.1184` (vs `0.1129` for the clean sibling `schlussredaktion-v1`). Overall practice CER is `0.1456` (`1577 / 10829` edits; 29 collapsed lines).
2. **Weight-space comparison of $\Delta_{\text{sub}} = W_{\text{sub}} - W_{\text{par}}$ vs $\Delta_{\text{sib}} = W_{\text{sib}} - W_{\text{par}}$:**
   - In the output layer `O_18` (character-aligned across 267 shared rows), $\cos(\Delta_{\text{sub}}, \Delta_{\text{sib}}) = 0.8458$ for weights and $0.9647$ for biases, but $\Delta_{\text{sub}}$ has a $+0.1184$ blank-bias jump (`+0.8792 -> +0.9976`, compared to `+0.9356` in `sibling`) and $1.33\times$ stronger downward push on rare/umlaut character biases—a classic signature of CTC training under label noise.
   - In the deeper layers, $\cos(\Delta_{\text{sub}}, \Delta_{\text{sib}})$ drops from $0.58$–$0.81$ in `L_16` to $0.42$–$0.70$ in `L_14`, $0.34$–$0.67$ in `C_0`–`C_9`, and only $0.24$–$0.28$ in `L_12.layer.weight_ih/hh`. Thus, the mismatched Hertwig image–text pairs heavily perturbed `L_12` and the conv stack in a direction orthogonal to the clean Schlussredaktion adaptation.
3. **Plan for Round 0 (`program_01.py`, Passes 1–4):**
   - **Pass 1 (`C0`):** Verify baseline CER `0.1456` and collapsed count `29`.
   - **Pass 2 (`H1`):** Test the minimal 1-parameter output edit: restoring `O_18.lin.bias[0]` (CTC blank) to the parent's value `+0.8792`.
   - **Pass 3 (`H2`):** Test character-aligned output-layer repair (`O_18.lin.weight` and `O_18.lin.bias`) using the midpoint $0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$.
   - **Pass 4 (`H3`):** Test full-network character-aligned midpoint merge ($0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$) across all 34 tensors (`C_0`–`O_18`), leaving 6 passes for rounds 1+ to tune the interpolation coefficient $\alpha$, test layer-wise schedules, or blend with subject weights based on the empirical results.
