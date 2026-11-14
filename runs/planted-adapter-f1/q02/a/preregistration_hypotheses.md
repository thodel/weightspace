# Pre-registered hypotheses — q02, task (a), budget 6

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort as configured by the caller (not raised by me); tools used: Read (brief, pack, the four API files), Write (these two files), Bash only for `py_compile`. No pass run, no adapter loaded, nothing read outside the allowed files.

## What the pack shows

- `lora_alpha` = 256, r = 64, scale 4.0. Both siblings: alpha 128, scale 2.0.
- Every one of the 128 modules has ‖ΔW‖ / sibling-median between 1.60 and 1.85 (median ≈ 1.79); the total is 235 against 127 / 136. There is no module that stands out from that band in either direction; effective ranks and σ1 look like a normal depth profile (layer 0 and the late gate/q modules lower rank, mid layers ~30–48).
- C0: CER 2.50, 29 / 32 pages at the cap, 28 loops, and even the pages that did not overrun are at CER ≥ 0.77 — every page is bad, which points to a global defect, not a local one.
- Reading: the update the base receives is twice what the recipe produces (alpha doubled, 256 instead of 128; the B·A themselves are ≈0.9× the siblings, ordinary run-to-run variation). ratio ≈ 1.8 = 2 × 0.9.

## Hypotheses (one line each: edit class — prediction — if it fails)

- **H1 (main)** — `scale`, all 128 `lora_B` × 0.5 (= set alpha back to 128; config stays 256, the saved adapter is equivalent). Predicted: CER falls from ≈2.5 to below 0.25 (expected 0.05–0.15, i.e. −2.3 points of page CER with overrun), pages at the cap fall from 29 to ≤ 3. If refused by the keep rule: the defect is not (only) the global scale; nothing else in the pack is anomalous, so what remains is invisible to norm/rank statistics (a sign flip or permutation inside a module) and cannot be located with 4 passes — I would then try one gentler global factor (H1b) and otherwise freeze with no further edit.
- **H1b (only if H1 refused)** — `scale`, all `lora_B` × 0.7. Predicted: CER −0.5 to −1.5 if the over-scale is real but 0.5 overshot; if refused too: conclude the scale is not the defect, freeze.
- **H2 (data-driven, zero passes unless it fires)** — spectral outlier check against the two siblings computed inside the program (`directions()` per module on this adapter and on both siblings; per-module top-singular-value share σ1/‖S‖ and gap σ1/σ2). A module is flagged only if its share exceeds the siblings' mean by > 0.20 absolute and > 1.6× relative, or its gap is > 4 and > 3× the siblings' maximum. Edit: `add` on that module's B, delta = −(1−t)·u1(u1ᵀB), shrinking σ1 to the siblings' share (t = target/σ1), at most 3 modules in one edit. Predicted: if a module is flagged, CER −0.01 to −0.05 (a planted spike removed); I expect **nothing to be flagged** (the norm ratios are flat, a spike would have raised one of them). If flagged and refused: the outlier is natural variation, revert.
- **H3 (only if loops remain after H1: last kept state has ≥ 4 pages at the cap)** — `scale`, all `lora_B` × 0.8 (alpha 128 → ≈102). Predicted: small change, CER −0.02 to −0.1 and fewer pages at the cap; if refused: the residual loops are not scale-driven, the halved adapter is the final state. If kept and still ≥ 4 at the cap and a pass is left, one more step of 0.8 (same prediction); never more than two steps.

## Pass plan (≤ 6)

1. C0 baseline. 2. H1. 3–6. H2 if flagged (1 pass), H3 if loops remain (1–2 passes); H1b only in the branch where H1 was refused. Unused passes stay unused. Always ends with `freeze()`.
