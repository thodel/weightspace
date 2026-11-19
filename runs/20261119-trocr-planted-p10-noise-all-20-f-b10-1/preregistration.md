# Pre-registration, 20261119-trocr-planted-p10-noise-all-20-f-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `f`: Claude Fable 5.2 (`claude-fable-5-2`), arm (a2), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s06/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s06/f/program.py`, copied unchanged (sha256 `6dd3ac92f89403d49ddb89599d1ddedc0dda976d1ad8fb31d82afe4637ec671c`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p10-noise-all-20` = diffuse: Gaussian noise on every encoder and decoder weight matrix, each at a fraction of its own norm (tensor digest `9b489cbf08b1caea4b2127425921a238a70273e568ac928500892f8a03fd64a7`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

Model id: claude-fable-5-2. Settings: task (a), budget 10, WITH the donor; reasoning effort low; single session, no subagents.
Tools used: Read and Bash (cat/grep and a small python parse of the pack tables only), python -m py_compile. Read: PROGRAMMER_BRIEF.md,
wsp/ops.py, wsp/session.py, and the three files of runs/trocr-planted/s06/pack/. No model loaded, no pass run, no other pack or run read.

# Reading

- C0: CER 0.2426, 7 collapsed lines, length ratio 0.968; confusions are diffuse (dropped spaces 385, dropped/inserted e, n, r, s). No single
  token or character class stands out: a global loss of precision, not a local lesion.
- describe.md: no zeros, no non-finite values, no tensor with an odd norm.
- diff: no moved head (>3x median: none), no token row with cosine < 0.5, layer norms and biases in the healthy range.
- The signature: EVERY 2-D weight matrix of the encoder layers sits at relative distance 0.200-0.225 from the parent with norm ratio
  1.020-1.025, while the encoder biases and layer norms of the same layers moved only 0.001-0.018 (the child barely trained the encoder).
  A norm ratio of 1.02 at distance 0.20 is what an added component orthogonal to the weights gives (sqrt(1+0.04) = 1.02). The decoder
  matrices show the same floor (0.212-0.307, ratio 1.02-1.05) on top of real fine-tuning movement, and their biases are in the healthy range.
  The tied embedding (0.527), position tables and patch embedding look healthy.
- Diagnosis: Gaussian noise of about 20 % of the tensor norm added to all weight matrices of the transformer layers (encoder and decoder),
  nothing else. One defect, global.

# Repair idea

Noise cannot be subtracted, but the parent is a noise-free prior. Per tensor, the least-squares shrinkage toward the parent is
interpolate(alpha) with alpha = sigma^2 / d^2 (sigma = 0.20, d = measured relative distance to the parent, clipped to 1): where the child
did not really move (encoder: alpha 0.8-1.0) this is nearly a transplant and removes almost all the noise; where it moved (decoder ffn.out:
alpha about 0.45) it keeps part of the child's learning. Alpha is computed in the program from the tensors, not from a pass.

# Hypotheses (edit class; prediction on the practice set, C0 = 0.2426; conclusion if it fails)

H1. Encoder matrices (enc.NN.attn.{q,k,v,o}.weight, enc.NN.ffn.{in,out}.weight, 72 tensors) carry 20 % noise and the true child is close
    to the parent. Edit: interpolate toward the parent with the shrinkage alpha (class: donor interpolation / denoise). Prediction: CER falls
    by 6 to 12 points (to 0.12-0.18), collapsed lines fall. If it is not kept: fallback H1b, full transplant (alpha = 1) of the same tensors,
    predicted -5 to -12 points; if that fails too, the encoder distance is real learning, not noise, and I leave the encoder alone.
H2. Decoder attention matrices (dec.NN.{self,cross}.{q,k,v,o}.weight, 96 tensors) carry the same noise over a true movement of about
    0.07-0.11. Edit: shrinkage interpolation, alpha about 0.75-0.9. Prediction: a further fall of 2 to 6 points. If not kept: fallback H2b
    with half the alpha (predicted -1 to -4); if that fails, the decoder attention distance is mostly learning and stays.
H3. Decoder FFN matrices (dec.NN.ffn.{in,out}.weight, 24 tensors), true movement about 0.15-0.23. Edit: shrinkage interpolation, alpha about
    0.45-0.65. Prediction: a further fall of 1 to 4 points; the least certain of the three, since most of the child's learning for this hand
    sits here. If not kept: fallback H3b with half the alpha (predicted -0.5 to -2.5); if that fails, leave the FFN noisy.
H4. (only with passes left) If H1 was kept with alpha < 1: move the encoder matrices the rest of the way to the parent (transplant).
    Prediction: change within +-0.5 points, most likely NOT kept by the rule; a kept result would say sigma was underestimated.
H5. (only with passes left) A further 0.3 step of all decoder matrices toward the parent. Prediction: within +-1 point, probably not kept;
    a clear gain would say the noise is larger than 20 % in the decoder; a clear loss confirms the shrinkage was not too weak.

Overall prediction: final practice CER between 0.06 and 0.14 (a repair of at least 10 points), not the full health of an unnoised child
because the decoder noise can only be shrunk, not removed. If H1 to H3 all fail, the noise diagnosis is wrong and the program ships the
model unchanged. At most 9 passes are used (1 baseline + up to 8), every edit decided by s.keep.

