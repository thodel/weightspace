model id: claude-sonnet-5-5 | settings: reasoning effort low, no donor (pack_nodonor only), MASK = s.unseen_by() | tools: Read, Bash (ls/cat/grep/py_compile), Write; no model loaded, no pass run

Readings: C0 CER 0.2346, 7 collapsed lines, top confusion = space deleted (377), then dropped letters. describe.md shows one sharp anomaly:
enc.03.ffn.out.weight L2 31.0 / bias 5.4 against neighbours 61-67 / 10.7-11.4 (about half). Other candidates are weak (dec.11.norm_ffn.weight 67 vs ~30.6 elsewhere).

H1 scale enc.03.ffn.out.{weight,bias} x2.0 (undo a halving): CER down ~0.03 to 0.10 (encoder features are damaged; likely the largest effect). If it fails: the low norm is natural for this layer; drop it.
H2 scale dec.11.norm_ffn.{weight,bias} x0.6 (norm of 67 vs 30 in all other decoder norms): CER change -0.005 to +0.01, low confidence. If it fails: norm is natural for the last layer; the edit is undone.
H3 scale all other enc.NN.ffn.out.{weight,bias} x1.3 (a global damping of encoder FFNs, not visible in single-tensor norms): CER down ~0.01 to 0.03 if present; if fail try x1.15; if both fail, no global encoder FFN damping.
H4 scale all dec.NN.cross.o.{weight,bias} x1.3 (global damping of cross attention output; would show as dropped characters/spaces): CER down ~0.01 to 0.03 if present; if fail try x1.15; if both fail, no cross-o damping.
Not attempted (no visible signature in norms/zeros/non-finite): zeroed rows/heads, swapped rows, noise, embedding shifts. Zero share is 0 and nothing is non-finite everywhere, so zeroing is unlikely.
Conclusion rule: an edit is believed only if s.keep accepts it (interval below -0.005); otherwise treated as no defect there. Max 7 passes used of 10.
