Model: claude-sonnet-5-5 (Sonnet 5.5). Settings: reasoning effort low; tools used: Read, Write, Bash (py_compile only). Task (a), budget 10, no donor, MASK = s.unseen_by().

Reading of the pack: healthy fine-tune. C0 CER 0.2307, length ratio 0.952, deletions dominate (space -> nothing 1540, e/r/n/s/i -> nothing, only 130 insertions of e). The model under-writes: it drops spaces and letters. No tensor shows zeros or non-finite values. Outliers: dec.11.norm_ffn.weight (norm 67 vs ~31 elsewhere), enc.00/enc.10 norm_ffn weights. No token list or tokenizer is available to me, so I make no token-specific edits.

Edits are uniform scalings (no direction guessed), applied one at a time; each is decided by the keep rule. Expected outcome for all: nothing to gain, |delta CER| < 0.3 points, rejected.

H1 (scale, all dec.NN.cross.o weights and biases x1.05): more weight on the image evidence relative to the language prior should reduce omissions. Predicted CER change -0.2 to -0.5 points if the under-reading comes from a prior-dominated decoder; else +0.1 to +0.5. If it fails with CER up: try x0.95 (branch). If CER down but not significant: try x1.10.
H2 (scale, dec.embed.tokens x1.04): the tied readout is sharper; the ranking of beams (shorter vs longer) shifts slightly. Predicted -0.1 to +0.3. The input side is almost undone by the embedding LayerNorm. If CER rises: try x0.96.
H3 (scale, enc.norm_final.weight and bias x1.03): larger encoder output increases cross-attention key/value magnitude, sharpening attention to the image. Predicted -0.1 to +0.3. If it fails: nothing further on the encoder.
H4 (scale, dec.11.ffn.out weight and bias x1.05): late-layer FFN contribution up; the odd norm_ffn of layer 11 hints at a large late contribution that the fine-tune tuned, so I expect a rise. Predicted +0.2 to +1.0 (so only tested if budget remains; revert expected).

Conclusion if all fail: the model sits at a local optimum for scalar rescalings; report no gain, keep the unedited weights.
