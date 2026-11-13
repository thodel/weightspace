# Activation atlas: report (epic thodel/weightspace#13)

Levels L0 (activation map) and L2 (dependence on the image) first, as the epic orders them; Gate 1 decides
whether the viewer may claim *where* in a line something happens. L1, L3 and L4 are not started.

## Gate 1 rule (fixed 13.11.2026, before any occlusion run)

Written and committed before `wsp.atlas` was run on any line, so that the result cannot shape the rule.

**Models and lines.** Practice lines the model was not trained on:

- `kraken-german-xix-blb-v1` on `blb-val-300` (no line seen by it);
- `kraken-bern-hertwig-v2` on the lines of `bern-val` whose `seen_by` does not name it (301 of 513).

From each set, lines of at least 15 characters, in the order of `sha256("20261113:" + key)`, the first 80
the transform can scale. No filter on CER: the sample is not chosen by how well the model reads.

**Occlusion.** A window 16 px wide (the full line height) is set to the line's median pixel value (the
background) and slid over the line with a step of 4 px. For each reference character *i* with forced CTC
run (*a*, *b*) (frames, from `wsp.layerview.forced_align`) and label *l*, its score is
`s_i = max over t in [a−2, b+2] of log p(t, l)`. Its occlusion profile is the drop `s_i(line) − s_i(occluded)`
per window position, and its *occlusion peak* is the window centre (px) with the largest drop.

**Hit.** A character is a hit when its occlusion peak lies nearer to the centre of its own CTC run
(`(a + b + 1) / 2 × px_per_frame`) than to the centre of any other character's run in the line.
A character whose largest drop is below 0.1 nat counts as a miss.

**Characters scored.** Non-space reference characters that the greedy reading gets right (`ok` in
`wsp.layerview.char_status`). Spaces (no ink to occlude), substituted and deleted characters are reported
separately and do not enter the gate.

**Pass.** Gate 1 passes for a model when the hit share is at least 0.80 and the lower end of its page-clustered
95 % bootstrap interval (`wsp.metrics.page_of`, 2,000 resamples) is at least 0.75. If it passes for both
models, the viewer may draw a character's location from the CTC alignment. If it fails for a model, that
model's viewer shows activation only and claims no location (the epic's rule).

**Reported alongside, not gating.** The signed offset (occlusion peak − CTC centre, px); the measured receptive
field of every block (relative change of the block's frame activation by the offset of the window), against
the receptive field computed from the VGSL spec and against a randomly initialised network of the same spec;
gradient × input of `s_i` summed per window, and how often its peak falls on the same character as the occlusion
peak.
