Model: claude-sonnet-5-5. Settings: low reasoning effort; tools Read, Write, Bash (py_compile only). Task (a), budget 6.

Evidence: lora_alpha 256 vs siblings 128 (scale 4.0 vs 2.0); every module's norm ratio is a uniform 1.7-1.85, total 235 vs 127-136; no module stands out. C0 CER 2.50 with 29/32 pages at the cap and 28 loops.

H1 (scale, all 128 B x0.5, i.e. alpha 256 -> 128). Predict CER falls from ~2.5 to ~0.3-0.6 and at_cap falls from 29 to under 10. If it fails (not kept), the update is not simply over-scaled; try a milder x0.7 and conclude the damage is not a pure global gain.
H2 (after H1 kept: fine scale x1.1, net 0.55, matching the norm ratio 1/1.8). Predict a change of at most +-0.05 CER, within noise; keep only if the rule says so. If it fails, net 0.5 stays.
H3 (after H2 fails: scale x0.85, net 0.425). Predict a change of +-0.05, likely not kept. If it fails, the 0.5 point is near the optimum.
H1b (if H1 not kept: x0.7 from the original, then x0.5 of that). Predict a CER fall of at least 1 point if the global-gain hypothesis is partly right. If both fail, I conclude the defect is not a global scale, nothing is shipped.
Null: if the repairs give no change beyond noise, ship the original adapter (nothing kept).
