Model id: claude-sonnet-5-5. Settings: low reasoning effort; tools used: Bash (read pack, grep/awk scan of diff table, py_compile), Read, Write. No model run.

H1: six rows of the tied embedding dec.embed.tokens are exactly zero (ids 6 ',', 4 '.', 90 't', 225 'en', 364 ' e', 2432 ' und'; cosine 0, norm 0). These tokens can be neither read nor written; matches C0 confusions (deleted '.', ',', 'e', space-before-word, 'n'). Edit: set those rows to the donor rows (the child's rows move ~0.5 relative elsewhere, so the donor row is the closest available estimate). Predicted: CER drops clearly (several points, perhaps 3-8).
No other anomaly: all other tensors within healthy distance, norm ratios 0.99-1.06, no head outliers, no non-finite. Nothing else applied.
