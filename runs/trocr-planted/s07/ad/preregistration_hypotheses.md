# Preregistration (programmer, task a, budget 10, with donor)
Model: claude-sonnet-5-5 (Sonnet 5.5). Settings: low reasoning effort; tools used: Read/Bash on the pack, brief and the allowed wsp API files only; no model loaded, program not run.

Readings: C0 CER 0.2320, 9 collapsed lines. Diff to the donor: `dec.05.cross.o.weight` and `dec.05.cross.o.bias` have relative distance 2.0, norm ratio 0.0 (both are exactly zero). No other tensor is outside the healthy-child range (matrices 0.03-0.23, embedding 0.53, no zero token rows, no outlier heads).

H1 (main). Decoder layer 5 cross-attention output projection was zeroed, so layer 5 never reads the encoder. Edit: interpolate both tensors to the donor with alpha 1.0 (= donor copy). Predicted: CER falls by 0.03 to 0.10 absolute (to about 0.13-0.20), collapsed lines fall. If it fails the keep rule: the donor tensor does not match this child's other layers; fall back to H2.
H2 (fallback, only if H1 reverted). Same tensors, alpha 0.6 (partial donor, since the child moved from the donor in the neighbours). Predicted: CER change -0.01 to -0.05. If it also fails: conclude the zeroing is not the cause of the damage on this set, or the donor tensor is incompatible; stop and make no further edits.
H3 (only if H1 kept; edit class scale). Scale factor 1.2 on the restored tensors (the child's healthy layers have bias norm ratio about 1.05); predicted change -0.003 to +0.003, expected to be reverted; a kept result would mean H1 under-restored. 
H4 (only if H1 kept and budget remains). Nothing else is flagged; conclude no further defect, apply nothing. Null expectation for any untargeted edit: no change.
