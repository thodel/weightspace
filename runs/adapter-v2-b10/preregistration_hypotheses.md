# Preregistration, programmer, adapter qwen3.5-4b-german-kurrent-xvi-xix-page-v2, run b10

Model id: claude-sonnet-5-5 (Claude Code, Agent SDK subagent). Settings: default effort, no tools beyond Read/Write/py_compile; I read only the pack, the brief and the harness API files; no model loaded, no program run. No sibling weights are used (donor rule moot).
Practice set: 32 pages. C0 reference: CER 0.729, truncated CER 0.333, 8 pages at the cap, 9 loops. Keep rule: paired bootstrap 95 % interval of CER(after) - CER(last kept) entirely below 0 and at-cap not up.
Plan: 1 baseline pass + up to 7 edit passes = at most 8 of 10 passes. Each kept edit becomes the new reference for the next. Expected outcome: nothing kept is a live possibility; I do not expect more than one or two edits to pass.

Note on the evidence: 4 of the 8 at-cap pages are short pages (reference 129 to 222 chars) that run on with the star token or quotes; the rest are long pages that repeat a clause. The lens shows the largest late directions mostly lower <|im_end|> (L31 o_proj, L31/L27 down_proj), but L30 down_proj dir 0 raises it and also promotes the star-token bytes (`Ġâľ`), so a stop switch and a loop writer may share a direction. Pages flipping loop/stop is chaotic; a one-page change (about 0.02 to 0.5 of mean CER, since looping pages have CER 1 to 16) can dominate the mean. I therefore only count effects over several pages.

## H1 (E1): remove end-of-turn suppressors, project_out
Edit class project_out (B <- (I - uu^T)B, axis 1), modules layers.31.mlp.down_proj and layers.27.mlp.down_proj, direction 0 (z(im_end) -3.32 and -2.80; no text token in the top-5 suppression of these).
Prediction: CER 0.73 -> about 0.60 (change -0.1 to -0.2, driven by 2 to 3 pages leaving the cap), truncated CER about unchanged (+-0.02), pages at the cap 8 -> 5 or 6.
If it fails: the lens z-score on im_end does not translate into stopping, or these directions are needed for ordinary reading (truncated CER rises); I conclude the loops are not produced by an end-of-turn suppressor in these modules.

## H2 (E2): strengthen the end-of-turn raiser, add
Edit class add (B <- B + 0.5 u u^T B, i.e. that direction x1.5), module layers.30.mlp.down_proj direction 2 (z(im_end) +2.27; deliberately not direction 0, which also promotes the star-token).
Prediction: CER -0.03 to -0.10, truncated CER about unchanged, pages at the cap -1 to -3.
If it fails: this direction is not a stop switch or the failure is not a missing stop signal.

## H3 (E3): late-MLP group scale, scale
Edit class scale, factor 0.9, B of gate/up/down_proj in layers 24-31 (24 modules).
Prediction: CER between -0.08 and +0.02 (probably flat), truncated CER 0.00 to +0.02, pages at the cap -1 to -2.
If it fails: the late MLP update is not where the loop is written, or any damping costs more reading than it saves.

## H4 (E4): whole adapter x0.9, scale
Edit class scale, factor 0.9 on all 128 B matrices. Norms are 0.80 to 0.93 of the siblings, so I do not expect an overshoot; this is the cheap test of "the loop comes from ΔW's strength".
Prediction: CER 0.68 to 0.76 (little net change), truncated CER +0.01 to +0.03 (reading degrades slightly), pages at the cap 8 -> 6 to 8.
If it fails (CER up, cap up): moving toward the base does not remove the loop; the base may loop too (unknown), and the loop is not an overshoot of the update.

## H5 (E5, branch)
If E4 kept: scale 0.9 again on all B (cumulative 0.81); prediction CER a further -0.02 to -0.06, truncated CER +0.01 to +0.03, cap -1; failure means the trend turned over.
If E4 not kept: scale 1.1 on all B (stronger learned stop); prediction CER 0.70 to 0.75, truncated CER -0.01 to -0.03, cap 8 -> 6 to 8; failure means no global strength fixes the loop.

## H6 (E6): layer-27 q_proj, scale
Edit class scale, factor 0.8, module layers.27.self_attn.q_proj (most concentrated module: effrank 17, sigma1 8.2, energy share 1.36).
Prediction: CER -0.05 to +0.03, truncated CER about unchanged, pages at the cap 8 -> 6 to 8.
If it fails: a layer-27 attention rewrite is not the loop mechanism.

## H7 (E7, only if E1 kept)
Edit class project_out, layers.31.self_attn.o_proj direction 0 (z(im_end) -3.60, but z(¬) -8.06 and it suppresses the star-token). The risky member of H1's family.
Prediction: CER a further -0.02 to -0.08 if stopping dominates, +0.05 or more if ¬ and the star-token return; cap -1 to -2.
If it fails: the suppression of ¬/star by this direction is needed for the reading. Skipped if E1 was not kept.

## Falsification overall
No edit kept means: no single direction, module group or global scale I tried removes the loop on the practice set beyond the one-page-flip noise; I will report "nothing found, nothing kept", not an invented gain.
