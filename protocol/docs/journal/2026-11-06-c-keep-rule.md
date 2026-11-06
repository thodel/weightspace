# 2026-11-06 · Keep rule gets a minimum effect

**Type:** protocol amendment · **Source:** commit `ee1ffd6`

The phase 1 session kept two edits whose practice effects were 0.016 and
0.008 CER points, both deep inside their bootstrap intervals. With a budget of
10 passes this rule selects noise.

**Amendment [A1](../amendments.md#a1):** keep only if the paired 95 %
interval against the last kept state lies entirely below −*min_effect*
(preregistered, default 0). The harness now refuses such a keep, undoes the
edit and logs the reason (`interval contains 0`, `CER rose`,
`no pass since the edit`); every pass is logged against both the baseline
and the last kept state.

Written before the next session; the phase 1 session is marked as run under
the earlier rule.
