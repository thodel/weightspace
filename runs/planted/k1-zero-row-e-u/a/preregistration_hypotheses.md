# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read and bash on the pack and the brief only, no model run.

Reading: output row `e` has norm 0.00 (all other rows 7.7 to 24), so the logit of `e` is its bias alone. `e` is the most frequent German letter; 910 of its occurrences are read as nothing, the rest as a/i/o/r/c/u. CER 0.162, length ratio 0.90. Conv and LSTM blocks look ordinary. The donor (other hand, other features) cannot supply a usable `e` row; the row has to be rebuilt from the model's own rows.

H1 (set_rows, O_18 row e): the `e` row was zeroed. Rebuild it as the unit direction of (count-weighted mean of rows a,i,o,r,c,u minus the mean of all other rows), norm 6. Predict CER 0.162 -> 0.10 to 0.13 (e deletions are 910 of 2088 edits; a rough row recovers part). If it fails (no CI below 0): the proxy direction is wrong; conclude the e feature is not in the span of confuser rows, try H2/H5.
H2 (set_rows, same row): add 0.5 x unit blank row to the direction (e frames are currently decided as blank). Predict a further 0.005 to 0.015 CER gain over H1, or a loss if e starts to eat blanks (collapse/insertions). If not kept: the pure confuser direction stays.
H3 (set_rows, norm x1.5, then x1.5 again): row norm 6 is below the typical frequent-letter norm of 8; louder e should recover more of the deletions. Predict 0.005 to 0.02 gain per step; stop when not kept (insertions of e rise).
H4 (add, O_18 bias at e, +1 then -1): the bias is whatever training left; with a rebuilt row a shift of 1 logit trades e deletions against e insertions. Predict at most 0.01 either way; failing means bias is fine.
H5 (set_rows, fallback if H1 not kept): e row = 0.6 x mean of rows a,o,u (e's most common wrong readings, similar frames). Predict 0.01 to 0.03 gain, weaker than H1; failing means leave the model with the zero row only and report the defect as found but not repaired by this budget.
No other defect is suspected from the readings (no collapse beyond 9 lines, conv/LSTM statistics regular), so no other edits are tried. Passes used at most 9 of 10.
