Model id: claude-fable-5-2. Settings: reasoning effort low; task (a), budget 10, WITH the donor; MASK = s.unseen_by("trocr-kurrent-XVI-XVII").
Tools used: Read and Bash (cat/grep/awk over the pack files, the brief, wsp/ops.py and wsp/session.py signatures; py_compile). No model loaded, no pass run.

# Readings
- C0 practice: CER 0.2346, 7 collapsed lines, dominant confusion space -> nothing (377), otherwise diffuse deletions/insertions: a general
  degradation of the visual reading, not a token-specific fault.
- Diff to the parent: exactly one outlier. `enc.03.ffn.out.weight` norm ratio 0.5027, cosine 0.9961, rel. distance 0.667;
  `enc.03.ffn.out.bias` norm ratio 0.5000, cosine 1.0000, rel. distance 0.6667. Every other weight matrix has norm ratio 1.00-1.03 and distance
  0.07-0.24 (healthy range); per-layer norm of enc.03 (127.9) is the dip of the encoder profile. No zero rows, no non-finite values, no
  outlying attention head, no embedding row with cosine < 0.5; the tied embedding moved by 0.527 (healthy is about 0.5).

# Hypotheses
- H1 (edit class: scale; the only edit I intend to ship). The output projection of the FFN of encoder layer 3 (weight and bias) was multiplied
  by 0.5. Edit: scale `enc.03.ffn.out.weight` and `enc.03.ffn.out.bias` by 2.0. Prediction: CER falls strongly, from 0.235 to roughly
  0.05-0.10 (a drop of 0.13 to 0.18), collapsed lines to 0-1, the space -> nothing confusion falls below 100; the keep rule accepts it.
  If it fails (CER not lower by more than 0.005): the halving is not the functional defect or is compensated elsewhere; I then try H1b.
- H1b (edit class: transplant/interpolate; only if H1 is not kept). Replace the bias by the parent's and set the weight to the subject weight
  rescaled to the parent's norm (factor = ||parent|| / ||subject||, about 1.99). Prediction: same as H1 within noise (it is nearly the same
  edit); if this also fails I conclude the readings do not locate a repairable defect in this tensor and ship nothing.
- H2 (no edit). After H1 nothing else is wrong: no further edit is planned. If after H1 the CER is still above 0.15, I note it as evidence of a
  second defect invisible in the norm/cosine readings (for example a permutation or noise below the healthy-movement range) and stop, since
  no reading tells me where to edit.
