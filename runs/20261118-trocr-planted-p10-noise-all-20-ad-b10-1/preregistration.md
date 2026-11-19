# Pre-registration, 20261118-trocr-planted-p10-noise-all-20-ad-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `ad`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s06/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s06/ad/program.py`, copied unchanged (sha256 `298568cf649fa78f8c8fabb0755e90b78baef3dd916f1306a9cb37754a9f19a1`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p10-noise-all-20` = diffuse: Gaussian noise on every encoder and decoder weight matrix, each at a fraction of its own norm (tensor digest `9b489cbf08b1caea4b2127425921a238a70273e568ac928500892f8a03fd64a7`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

# Preregistration, task (a), budget 10, with donor (pack s06)

Model id: claude-sonnet-5-5 (Sonnet 5.5). Settings: reasoning effort low; tools used: file read, shell (grep/awk on the pack, `python -m py_compile`).
No model loaded, nothing run. Files read: brief, pack (describe.md, c0_practice.md, diff_parent_to_subject.md), signatures in wsp/ops.py and wsp/session.py.

## Reading of the pack
- No non-finite values, no zero share anywhere, no zeroed rows (embedding rows: 0 with cosine < 0.5, 0 zero rows), no attention head over 3x the median.
- Per-tensor distance to the parent is uniform: weights 0.20-0.31, layer-norm weights 0.003-0.004, biases 0.002-0.19; norm ratios 0.98-1.06; no layer stands out. Embedding 0.527 distance, cosine 0.861 (healthy step is about 0.5).
- C0 CER 0.2426 with 7 collapsed lines; dominant error is a deleted space (385) and deleted letters (e, n, s, r), plus n->r. This looks like a hard hand rather than a planted fault; the parent is said to read this hand worse.
- Main hypothesis H0: the model is healthy (null). Expect no edit to pass the keep rule. Edits below are low-prior probes, each moderate and reversible, kept only if the harness rule passes.

## Hypotheses (one edit each, in order; C0 is pass 1; at most 5 passes used)
- H1, interpolate `dec.embed.tokens` toward the donor, alpha 0.2 (tests a damaged/noisy embedding that the cosine 0.86 could hide). Predicted: CER change within +-0.3 points (about 0); if the embedding were damaged, -1 point or more. If it fails: embedding is not the problem; revert.
- H2, scale all decoder `cross.o` weights and biases by 1.05 (tests an under-weighted encoder signal, to explain the many deleted characters and spaces). Predicted: -0.3 points if true, +0.5 or worse if wrong (norm ratio of cross.o bias is already 1.05). If it fails: cross-attention strength is not the problem; revert.
- H3, interpolate encoder layers 10 and 11 (all weight and bias tensors) toward the donor, alpha 0.25 (tests a defect in the last encoder layers that the uniform distances would not show). Predicted: +0.3 points (donor reads this hand worse) under H0; -1 point or more if a localized defect exists. If it fails: revert; no encoder defect.
- H4 (only if H1 was kept): extend H1 to alpha 0.4. Predicted additional -0.3 points. If it fails: revert, stay at alpha 0.2.
- Collapsed-line rule: any edit that raises collapsed lines above 12 is reverted at once regardless of CER.
- Overall: if nothing is kept, I conclude the model is healthy and ship the unchanged C0 state.

