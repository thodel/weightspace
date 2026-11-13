Model: claude-sonnet-5-5. Settings: low reasoning effort; tools used: Read, Bash (reading pack and API files only; no model loaded, no passes).

## H1 (applied): the CTC blank bias is inflated by about +2
Evidence: practice errors are almost all deletions (reference -> nothing: space 177, e 81, n 53, i 38, r 37 ...), 14 collapsed lines,
length ratio 0.941. Blank bias +3.232 against -1.9 +- 2.0 for other rows; parent-to-child blank bias shift +2.113, while every other
aligned row's bias moves little or negatively. A healthy fine-tune would move the blank bias by a few tenths, not 2.
Edit: add -2.0 to `O_18.lin.bias[0]` (class: output-bias add). Predicted: CER falls from 0.079 to roughly 0.03-0.05, deletions drop sharply, collapsed lines near 0.
If wrong (the shift were real training): CER would rise modestly with more insertions; -2.0 is the full anomaly, so some risk.

## Not applied
- C_3.co.bias rel L2 1.09 vs parent: large but its L2 is small (0.10) and C_3 weights moved moderately; no behavioural symptom. Not touched.
- Row norms, the new-character rows, LSTMs: nothing anomalous seen.
