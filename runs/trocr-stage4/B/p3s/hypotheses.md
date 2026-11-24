Model: claude-sonnet-5-5 (Sonnet 5.5). Settings: default reasoning effort (not set by me); tools used: Read, Write, Bash (py_compile only). No donor; MASK = s.unseen_by().
Task (a), budget 10, subject B (healthy fine-tune; nothing planted). Prior: most likely nothing clears the keep rule (-0.5 CER points, page-clustered). The practice reading shows deletions dominating
(space 660, e 405, n 228, r 206, s 195, i 187 dropped; 300 spaces inserted; length ratio 0.958; 34 collapsed lines), so the hypotheses are about the model writing too little.
Pass 1 is the baseline C0. Each hypothesis below is tried once at a modest size; if kept it is repeated once at the same size, if refused the opposite size is tried once. Max 9 edits, 10 passes.

H1. Output temperature (scale dec.11.norm_ffn.{weight,bias} by 1.1, the final LayerNorm gain, which is 2x larger than in other layers and scales every logit). Class: scale (global logit temperature).
    Prediction: CER change between -0.3 and +0.3 points (beam ranking barely changes; only the EOS/length trade-off and token-vs-space competition shift). A sharper distribution may reduce early drops. If it fails, try 0.9; if both fail, conclude the readout temperature is already tuned by the fine-tune.
H2. Sharper cross-attention in the upper decoder (scale dec.06-11 cross.q weight+bias by 1.1). Class: scale (attention temperature). Prediction: -0.2 to +0.5 points. Rationale: deleted characters suggest the decoder
    skips over some image positions; sharper focus might lock onto the next glyph. If it fails, try 0.9 (softer, mixing neighbours); failure of both means the attention is at its optimum.
H3. Encoder output gain (scale enc.norm_final.{weight,bias} by 1.05). Class: scale. Prediction: -0.2 to +0.3 points; scales cross-attention keys (sharper) and values (stronger image evidence vs. language-model prior) together.
    If it fails try 0.95. Failure of both: the image/language balance is not a lever worth the size tried.
H4. Image evidence in the decoder residual (scale dec.07-11 cross.o weight+bias by 1.1): more weight on the image, less on the language prior, against dropped characters. Class: scale.
    Prediction: -0.3 to +0.5 points; a fail (CER up) means the language prior is helping and the deletions are not prior-driven.
H5. End-of-sequence row (dec.embed.tokens row 2 scaled by 0.9): the tied row is the EOS readout and also the start token input, so side effects are possible. Class: set_rows. Prediction: -0.3 to +0.3 points
    (fewer early stops would lift length ratio from 0.958); if it fails (CER up or collapses rise) revert; no opposite attempt, since the length ratio is below 1.

Conclusion rule overall: if no edit is kept, the result is "nothing to gain with global scalings of these sizes"; the final model is then the unedited one (the harness reverts every refused edit).
Expectation: probably no gain; at most one marginal keep, which would itself be weak evidence on unseen lines.
