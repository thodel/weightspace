# Pre-registration — 20261126-planted-k10-noise-lstm14-60pct-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k10-noise-lstm14-60pct-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s11/b/program.py`, copied unchanged.

1. **Subject:** `k10-noise-lstm14-60pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 60 % of the block's; `runs/planted/plant.json`) — tensor digest `930c5c2b0279e885f4e87954711d76aaee3e9d07128fc217156821da237eb2fc`, file sha256 `c55386f265d20e1ba0952391b63ff5e071a8277ec4544cbe9b0a86e9453f2bad`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — s11 task (b) (budget 0, blind programmer)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file` (with `py_compile`), `finish_round` (available: `read_file`, `list_dir`, `write_file`, `finish_round`).

## Diagnosis from pack readings

1. **Isolated defect in `L_14` weight matrices (`weight_ih_l0`, `weight_hh_l0`, `weight_ih_l0_reverse`, `weight_hh_l0_reverse`):**
   - In `diff_parent_to_subject.md`, the adjacent BiLSTM blocks `L_12` and `L_16` show normal fine-tuning drift from the parent (`kraken-german-xix-v1`):
     - `L_12`: `weight_hh` rel L2 `0.2079` / `0.2111` (cosine `0.9788` / `0.9781`), `weight_ih` rel L2 `0.2980` / `0.2946` (cosine `0.9556` / `0.9567`), `bias_hh` rel L2 `0.0419` / `0.0414` (cosine `0.9992`).
     - `L_16`: `weight_hh` rel L2 `0.1985` / `0.2188` (cosine `0.9806` / `0.9763`), `weight_ih` rel L2 `0.2964` / `0.3267` (cosine `0.9563` / `0.9467`), `bias_hh` rel L2 `0.0461` / `0.0441` (cosine `0.9990`).
   - In `L_14`, the biases `L_14.layer.bias_hh_l0` and `L_14.layer.bias_hh_l0_reverse` are completely healthy (rel L2 `0.0327` and `0.0317`, cosine `0.99951`), as are all convolutional blocks (`C_0`..`C_9`) and the output layer (`O_18`, 225 shared rows mean rel L2 `0.1930`, blank rel L2 `0.2253`, plus 72 new child codec rows).
   - However, the four weight matrices of `L_14` show anomalous relative L2 changes, reduced cosines, and inflated L2 norms in `describe.md`:
     - `L_14.layer.weight_hh_l0` (`[800, 200]`, $N = 160{,}000$): L2 `137.77`, rel L2 `0.5483`, cosine `0.87438`, abs max `2.063`.
     - `L_14.layer.weight_hh_l0_reverse` (`[800, 200]`, $N = 160{,}000$): L2 `133.93`, rel L2 `0.5572`, cosine `0.87110`, abs max `1.875`.
     - `L_14.layer.weight_ih_l0` (`[800, 400]`, $N = 320{,}000$): L2 `148.87`, rel L2 `0.7520`, cosine `0.78628`, abs max `1.863`.
     - `L_14.layer.weight_ih_l0_reverse` (`[800, 400]`, $N = 320{,}000$): L2 `149.60`, rel L2 `0.7400`, cosine `0.79233`, abs max `1.429`.
   - Solving the law of cosines $r^2 = 1 + \rho^2 - 2\rho\cos(P, C)$ for $\rho = \|C\| / \|P\|$ yields parent norms $\|P\| = 121.96, 118.07, 122.53, 123.52$ and excess squared norms $\|C\|^2 - \|C_0\|^2 \approx 3600$ for both `[800, 200]` matrices and $\approx 7200$ for both `[800, 400]` matrices. This matches i.i.d. zero-mean additive noise of standard deviation $\sigma = 0.15$ ($\sqrt{3600 / 160000} = \sqrt{7200 / 320000} = 0.150$) injected into the four `L_14` weight tensors, degrading baseline practice CER to `0.1200` (with 11 collapsed lines and generic character deletions).

## Shipped Edit (H1)

- **Hypothesis (H1):** The four weight matrices of the middle BiLSTM layer (`L_14.layer.weight_ih_l0`, `L_14.layer.weight_hh_l0`, `L_14.layer.weight_ih_l0_reverse`, `L_14.layer.weight_hh_l0_reverse`) are corrupted by additive isotropic noise ($\sigma \approx 0.15$), while `L_14` biases and all other blocks (`C_0`..`C_9`, `L_12`, `L_16`, `O_18`) are intact. Because `L_12`, `L_14`, and `L_16` share a tightly aligned hidden-state representation between parent and child (bias cosines `0.9990`–`0.9995`, uncorrupted weight cosines `0.95`–`0.98`), transplanting the four clean `L_14` weight matrices from the parent donor (`kraken-german-xix-v1`) eliminates the planted noise while preserving all child adaptations in the convolutional front-end, `L_12`, `L_14` biases, `L_16`, and the 297-class `O_18` output layer.
- **Edit class & operation:** `transplant` on `["L_14.layer.weight_ih_l0", "L_14.layer.weight_hh_l0", "L_14.layer.weight_ih_l0_reverse", "L_14.layer.weight_hh_l0_reverse"]` using donor tensors from `/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel` via `per_tensor`.
- **Predicted effect:** Practice CER (`0.1200` at C0) will decrease substantially by approximately `0.04` to `0.06` (down to ~`0.06`–`0.08`), collapsed lines will drop from 11 toward 0, and deletion errors (` `, `e`, `n`, `r`, `i` → `∅`) will be sharply reduced.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s11/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 172904 input / 2843 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
