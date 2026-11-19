# Pre-registration, 20261118-trocr-planted-p03-cross-o5-zero-ad-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `ad`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s07/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s07/ad/program.py`, copied unchanged (sha256 `37a923d06740aa44000b6c1456e0790742b78297e4b8a608bf6a746f687ed3c1`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p03-cross-o5-zero` = decoder layer 5 cross-attention output projection (weight and bias) x0.5 (tensor digest `0cd9334534d039638da99733c2964c80bd6ba017010a899024ba8d23b668f438`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

# Preregistration (programmer, task a, budget 10, with donor)
Model: claude-sonnet-5-5 (Sonnet 5.5). Settings: low reasoning effort; tools used: Read/Bash on the pack, brief and the allowed wsp API files only; no model loaded, program not run.

Readings: C0 CER 0.2320, 9 collapsed lines. Diff to the donor: `dec.05.cross.o.weight` and `dec.05.cross.o.bias` have relative distance 2.0, norm ratio 0.0 (both are exactly zero). No other tensor is outside the healthy-child range (matrices 0.03-0.23, embedding 0.53, no zero token rows, no outlier heads).

H1 (main). Decoder layer 5 cross-attention output projection was zeroed, so layer 5 never reads the encoder. Edit: interpolate both tensors to the donor with alpha 1.0 (= donor copy). Predicted: CER falls by 0.03 to 0.10 absolute (to about 0.13-0.20), collapsed lines fall. If it fails the keep rule: the donor tensor does not match this child's other layers; fall back to H2.
H2 (fallback, only if H1 reverted). Same tensors, alpha 0.6 (partial donor, since the child moved from the donor in the neighbours). Predicted: CER change -0.01 to -0.05. If it also fails: conclude the zeroing is not the cause of the damage on this set, or the donor tensor is incompatible; stop and make no further edits.
H3 (only if H1 kept; edit class scale). Scale factor 1.2 on the restored tensors (the child's healthy layers have bias norm ratio about 1.05); predicted change -0.003 to +0.003, expected to be reverted; a kept result would mean H1 under-restored. 
H4 (only if H1 kept and budget remains). Nothing else is flagged; conclude no further defect, apply nothing. Null expectation for any untargeted edit: no change.

