# Hypotheses (task b, budget 0)
Model: claude-sonnet-5-5. Settings: default reasoning effort; tools: Read, Write, Bash (py_compile only).

H1. lora_alpha is doubled (256 vs siblings' 128; scale 4.0 vs 2.0). Every one of the 128 modules has norm ratio ~1.75-1.85 against the siblings (uniform, no outlier module), total 235 vs 127/136. The C0 reading is collapsed (CER 2.50, 29/32 pages at cap, 28 loops), as expected for an over-strong update.
Edit: scale all lora_B tensors by 0.5 (equivalent to lora_alpha=128, scale 2.0, the siblings' recipe).
Predicted: large drop in CER (from ~2.5 to roughly the siblings' regime, <0.3) and far fewer pages at the cap.

No other defect is visible: no per-module outliers, effective ranks look normal. No further edits shipped.
