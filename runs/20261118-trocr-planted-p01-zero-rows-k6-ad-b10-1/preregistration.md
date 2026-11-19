# Pre-registration, 20261118-trocr-planted-p01-zero-rows-k6-ad-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `ad`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s03/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s03/ad/program.py`, copied unchanged (sha256 `5ac4bcaf932425efbd629c9a72f9d4b4fb8b70c2abfecedb51b67782b93f8acf`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p01-zero-rows-k6` = the rows of the k most frequent token ids of the practice text set to 0 in the tied embedding (input and readout) (tensor digest `22b6a4c8bdabbc6fb8829ec7f60c743999ccb893f669d5a30b4724f9a488b0ca`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

# Preregistration, task (a), budget 10, with donor (pack s03)
Model id: claude-sonnet-5-5 (Sonnet 5.5). Reasoning effort: low. Tools used: Read/Bash on the brief, the pack files and wsp/ops.py signatures only; no program run, no model loaded.

Readings: the diff shows exactly 6 rows of `dec.embed.tokens` exactly zero (norm ratio 0, cosine 0): ids 6 `,`, 4 `.`, 90 `t`, 225 `en`, 364 ` e`, 2432 ` und`. All other tensors move within the healthy range (matrices 0.08-0.23, embedding 0.53 overall); no head outliers; no non-finite values. C0 confusions fit: space/e/./n/, deleted (319, 63, 63, 47, 33), CER 0.2508.

H1 (main). The 6 zeroed rows of the tied embedding are a planted defect (token never readable or writable). Edit: set_rows on `dec.embed.tokens`, ids [6,4,90,225,364,2432], values = donor rows. Prediction: CER falls by 3 to 8 points (0.2508 -> about 0.18-0.22), deletions of `.`, `,`, `e`, `n` and spaces drop. If it fails (change within -0.005): the zeroing is not the cause of the errors; conclude donor rows are a poor fit for this child (embedding moves about 0.5 in healthy children) and try H2.
H2 (fallback if H1 is not kept). Same rows, donor row scaled by 0.5 plus nothing else (interpolate toward zero would be the same as a scaled row): set_rows with 0.5 * donor rows. Prediction: smaller gain than H1 or equal; if kept and better than none, keep. Conclusion if fails: rows need a child-specific value, leave unrepaired.
H3 (after H1 kept). Remaining error is the child's natural error, but check: interpolate whole `dec.embed.tokens` (all rows) 0.15 toward donor. Prediction: change within +-0.3 points (no defect expected); expected to be reverted. If it improves by more than 0.5 points, the embedding is partly corrupted: conclude generic drift.
H4 (after H1 kept, optional). Decoder cross-attention output biases `dec.*.cross.o.bias` have norm ratio 1.05-1.06 (all layers, consistent), not outlier: expected healthy. Probe: interpolate all 12 toward donor alpha 0.5. Prediction: 0 +-0.3 points, revert. Conclusion if gain: bias drift is real; keep.
H5. Null elsewhere: no other defect (encoder, layer norms, ffn) is visible in the diff; I will not edit them.
Budget: C0 (1) + H1 (1) + at most H2 (1) + H3 (1) + H4 (1) = 5 passes of 10.

