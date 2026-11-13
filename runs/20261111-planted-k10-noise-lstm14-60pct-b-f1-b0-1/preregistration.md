# Pre-registration — 20261111-planted-k10-noise-lstm14-60pct-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k10-noise-lstm14-60pct-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s11/b/program.py`, copied unchanged.

1. **Subject:** `k10-noise-lstm14-60pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 60 % of the block's; `runs/planted/plant.json`) — tensor digest `930c5c2b0279e885f4e87954711d76aaee3e9d07128fc217156821da237eb2fc`, file sha256 `c55386f265d20e1ba0952391b63ff5e071a8277ec4544cbe9b0a86e9453f2bad`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — planted-f1 / s11 / task (b), budget 0 (blind programmer)

Model id: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort; tools used: Read/ls/wc on the
three pack files and the allowed `wsp/` sources, Write for the two files, `py_compile` only. No pass, no model loaded.

## What the readings show

- **Practice C0:** CER 0.120, WER 0.319, 11 collapsed lines, length ratio 0.955. The error profile is dominated by
  deletions: ` `→∅ 156 (by far the largest), then e/n/r/i/./s/t→∅. WER is high relative to CER, which is what
  swallowed word gaps do.
- **Output layer:** the bias of the space row is **−7.85, the most negative bias of all 297 rows**, below `ú`, `Ⅲ`,
  `Ž`, `Ⅱ` (characters that practically never occur). Non-blank biases are −1.92 ± 2.01; frequent letters sit near
  0 (`s` +0.40, `d` +0.29). The most frequent character of the hand cannot have trained its bias three sigma below the
  mean and below the never-seen characters. The space row is *not* in the "moved most" list of the parent diff, so its
  weight row is normal (rel L2 < 0.38): the plant is in the bias only.
- **L_14 (middle BiLSTM):** `weight_ih` rel L2 0.752/0.740 and `weight_hh` 0.548/0.557 against the parent, versus
  0.30/0.21 (L_12) and 0.30–0.33/0.20–0.22 (L_16). Yet `L_14.bias_hh` moved **less** than the neighbours' (0.033 vs
  0.042, 0.046). Training moves a layer's weights and biases together; 2.5× the weight change with the smallest bias
  change is not training. The cosines fit an orthogonal perturbation (predicted 0.799 / 0.877 for the observed rel L2,
  seen 0.786 / 0.874), the rms of L_14 grew 25 % (ih) and 14 % (hh) while the neighbours grew 2–4 %, and the implied
  per-element σ is the same for ih and hh (≈ 0.15): isotropic Gaussian noise of σ ≈ 0.15 was added to the four L_14
  weight matrices (both directions), biases untouched. Such noise in the middle recurrent layer is a plausible cause of
  the collapsed lines, the general under-emission (deletions of e/n/r/i/s/t, length ratio 0.955) and the CER level.
- **Seen, not acted on:** `C_3.co.bias` rel L2 1.09 / cos 0.68 — both vectors are tiny (L2 0.10 in the child, ≈ 0.07
  in the parent, per-element change ≈ 0.013) against a conv weight of L2 40; functionally negligible either way.
  `L_16.weight_ih_l0` abs max 4.70 — one element, no effect worth an edit. Convs, L_12, L_16, blank row (rel 0.23,
  bias Δ +0.11) and the new codec rows look like ordinary fine-tuning.

## H1 — space bias planted down (edit class: output-row bias repair, `set_rows` on `O_18.lin.bias`)

Hypothesis: the space row's bias was lowered by several units (to −7.85). Edit: set it to the parent's space bias plus
the median child−parent bias shift of the frequent lowercase letters (so it lands where normal fine-tuning would have
put it). Guard inside the program (free, reads weights only): apply only if the current value lies more than 3 below
that target; otherwise skip and note. Predicted effect: most of the 156 space deletions disappear, a handful of new
space insertions appear; CER down by roughly 0.008–0.011 (100–140 of 12,867 characters), WER down markedly
(0.32 → ~0.25). If the actual plant was a smaller shift, the model slightly over-emits spaces; still a net gain.

## H2 — Gaussian noise on the L_14 weight matrices (edit class: block interpolation toward the parent donor)

Hypothesis: `L_14.layer.weight_{ih,hh}_l0[_reverse]` = healthy child + N(0, 0.15²). Edit: `interpolate` the four
tensors toward the parent with α = 0.8 (biases untouched). This removes 80 % of the noise (residual rel ≈ 0.14, 4 % of
the noise energy) at the price of 80 % of the layer's genuine adaptation (≈ 0.3 rel, of which 20 % is kept); fine-tuning
drift lies in flat directions and random noise does not, so α in 0.8–1.0 is near-optimal if the hypothesis holds and
0.8 is the mild hedge if it does not. Guard inside the program: apply only if every L_14 weight tensor's rel L2 to the
parent exceeds 1.5× the largest rel L2 of the same kind of tensor in L_12/L_16 and L_14's bias_hh did not move more
than 1.2× the neighbours' average. Predicted effect: the largest gain of the session — collapsed lines 11 → ~0–3,
deletions of frequent letters down, CER down by 0.03–0.06 (to ~0.06–0.09). If it fails (CER up), the L_14 drift was
genuine and the noise reading was wrong.

Order in the program: H1 (space bias), then H2 (L_14); the tensors are disjoint, so the order does not matter for the
final weights. Nothing else is edited.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s11/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
