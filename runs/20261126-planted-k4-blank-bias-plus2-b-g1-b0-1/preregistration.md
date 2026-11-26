# Pre-registration — 20261126-planted-k4-blank-bias-plus2-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k4-blank-bias-plus2-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s03/b/program.py`, copied unchanged.

1. **Subject:** `k4-blank-bias-plus2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: blank bias +2.0; `runs/planted/plant.json`) — tensor digest `339161afc6ffa62ee51b2d9e5ce3dd0bbd0c3f3cd7939c6300bfa46500dc993b`, file sha256 `2727c5266842c02e67e61b779bf0d900f981af6137e12447ed7f279c833c268c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — pack s03, task (b) (budget 0)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`

## Diagnosis from pack readings

1. **Practice-set failure pattern (`c0_practice.md`):**
   - Baseline CER is `0.0789` (WER `0.2135`, `1015` character edits across 300 lines), with **14 collapsed lines** and a low length ratio of **`0.941`**.
   - All top-10 confusions are pure deletions (`char → ∅`), led by `' ' → ∅` (177), `'e' → ∅` (81), `'n' → ∅` (53), `'i' → ∅` (38), `'r' → ∅` (37), `'s' → ∅` (28), `'t' → ∅` (26), `'.' → ∅` (26), `'o' → ∅` (23), `'u' → ∅` (21). This is the signature of an over-biased CTC blank class suppressing character emissions across the board.

2. **Weight inspection (`describe.md` and `diff_parent_to_subject.md`):**
   - In `describe.md`, the CTC blank row (`row 0` of `O_18.lin`) has weight L2 norm `4.637` and an abnormally large positive bias of **`+3.232`** (whereas non-blank rows have mean bias `-1.921 ± 2.010`, and the next highest bias in the entire layer is `'s'` at `+0.40`).
   - In `diff_parent_to_subject.md`, all convolutional, recurrent, and output weight tensors show normal fine-tuning drift from the parent `kraken-german-xix-v1` (output weight rows mean relative L2 change `0.1930`; blank row weight relative L2 change `0.2253`). However, the blank row bias shifted by **`bias Δ = +2.113`** relative to the parent (from `+1.119` in the parent to `+3.232` in the subject), consistent with a planted `+2.0` additive defect on `O_18.lin.bias[0]` on top of normal fine-tuning drift (`+0.113`).

## Hypothesis H1

- **Hypothesis:** `O_18.lin.bias[0]` (the CTC blank logit bias) carries a planted positive offset of `+2.0` (raising it to `+3.232`, `Δ = +2.113` over the parent's `+1.119`), which causes widespread character deletions (`length_ratio = 0.941`) and 14 collapsed lines.
- **Edit (`add` on `O_18.lin.bias`):** Subtract `2.0` from `O_18.lin.bias[0]` (adding a delta vector with `delta[0] = -2.0` and `0.0` elsewhere, restoring `O_18.lin.bias[0]` from `+3.232` to `+1.232`).
- **Predicted effect:** CER will decrease substantially by approximately `0.030` to `0.045` (from `0.0789` down to `~0.035–0.048`), eliminating the 14 collapsed lines, restoring the length ratio to `~1.00`, and removing the dominant `char → ∅` deletion errors.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s03/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 113265 input / 1742 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
