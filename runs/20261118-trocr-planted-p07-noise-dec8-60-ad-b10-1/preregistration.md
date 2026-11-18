# Pre-registration, 20261118-trocr-planted-p07-noise-dec8-60-ad-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `ad`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s08/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s08/ad/program.py`, copied unchanged (sha256 `0d73791cc4334bf073b897c3f6b62df8f08afb9517fd33ac495a3f9644249b7b`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p07-noise-dec8-60` = Gaussian noise on the weights of decoder layer 8 (self, cross, FFN), total norm a fraction of the layer's (tensor digest `1547142fbf5cd16953eec8cec5b0de50535bd5fd1d294fefe00309925161d6a1`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

# Preregistration, session s08, task (a), budget 10, with donor

Model id: claude-sonnet-5-5 (Sonnet 5.5). Settings: reasoning effort low; tools: file read, shell (only for parsing the pack and `py_compile`); no model loaded, no pass run.

Reading of the pack: C0 CER 0.2321, 5 collapsed lines; most errors are deleted spaces (381) and deleted letters. Against the parent, only decoder layer 08 is abnormal:
all 11 weight matrices have rel. distance 0.37-0.77 (healthy 0.07-0.24), norm ratio 1.07-1.35 (healthy ~1.00-1.03), cosine 0.74-0.93 (healthy >0.97). Biases and norms of dec.08 are normal; the
tied embedding (0.53), other layers, encoder, and head table are normal. Norm inflation with cosine loss of this size is the signature of random noise added to dec.08 weights.

H1 (interpolate, dec.08 weight matrices, alpha 0.5 toward the parent): removes about half of the noise amplitude; predicted CER 0.232 -> about 0.17-0.19 (drop 4-6 points).
  If it fails (not kept): noise is not the defect or the parent direction is bad for this hand; I try alpha 0.25 once, then stop and report.
H2 (further interpolate, cumulative alpha 0.75): predicted further drop of 1-3 points (CER about 0.15-0.17). If it fails, stay at 0.5 and try cumulative 0.62 once.
H3 (cumulative alpha 0.9): predicted change -0.5 to +1 point; the child's own adaptation in dec.08 is lost as alpha ->1, so a plateau or small rise is expected. If it fails, keep H2 state.
H4 (norm correction, scale dec.08 weights so their norm matches the parent's; only if state is still alpha < 1): predicted -0.5 to -1 point. If it fails, noise is already removed enough; stop.
H5 (null control for stopping): if H1 and its fallback are both rejected, I conclude that dec.08 is not repairable by pulling toward the parent and ship no edit.
Not edited: embeddings, other layers, biases (within healthy range).

