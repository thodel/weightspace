# Notes — 20261106-german-xix-blb-v1-b10-1

Programmer: Claude Fable 5.1, Claude Code on asteraix. The whole plan was written into `program.py` before the sandbox run (preregistration.md §5); the sandbox ran unattended, so these notes are written after the run from `log.jsonl`, with the log's timestamps (UTC, 06.11.2026).

- 15:29:37 start. 4,111,961 parameters, budget 10. Pass 1 = C0: CER 0.06458, WER 0.1711, 10 collapsed, 831 edits on 12,867 chars; top confusions ` `→∅ 103, `e`→∅ 40, `n`→∅ 35.
- Pass 2, H1 (blank bias −0.113, back to the parent's 1.1195): CER 0.06443 (−0.016 points), WER 0.1703, collapsed 10, ` `→∅ 103 → 98. Kept by the rule; the paired bootstrap interval [−0.062, +0.032] points contains 0. This is noise-sized, not the predicted −0.1 to −0.5.
- Pass 3, H2 (blank −0.5 total): CER 0.06505, WER 0.1747, collapsed 9; deletions keep falling (` `→∅ 90) but insertions and substitutions rise faster. Reverted.
- Pass 4, H3 (space bias +1.0): CER 0.06474, ` `→∅ 103 → 90 as predicted, but `∅→ ` insertions take the gain back. Reverted. The space row's −7.85 bias is not the reason spaces are deleted; the frames where a space is dropped are frames where the blank wins against everything.
- Pass 5, H4 (space +0.5): CER 0.06489. Reverted. The lever is monotone in the wrong direction at both sizes.
- Pass 6, H5 (12 typographic rows from the parent): CER 0.06458, +0.00016 against the kept state, 0 against C0; the readings' null prediction holds (the rows are inert on this text). Reverted (a tie reverts).
- Pass 7, H6 (frequent rows ×1.1): CER 0.06505, collapsed 9; `e`→∅ 40 → 37 and `n`→∅ 35 → 31 as predicted, substitutions among frequent letters rise more. Reverted.
- Pass 8, H7 (blank −1.0 total, 1.2323 → 0.2323): CER 0.06435, WER 0.1755, collapsed 9, length ratio 0.982; ` `→∅ 73 but `∅→ ` 31 enters the top three. Kept by the letter of the rule (CER 0.06443 → 0.06435, collapsed 10 → 9) with WER up by half a point and an interval of [−0.34, +0.30] points. I would not call this a gain; the rule keeps it, the held-out score decides.
- Passes 9–10 unspent: every lever had been measured at two sizes, and spending them on a third size of the blank bias would have been curve-fitting the practice set.
- 15:32:30 freeze (173 s wall for 8 passes on 12 CPU threads). 7 edits, 2 kept, one tensor changed (`O_18.lin.bias[0]`, 1.2323 → 0.2323). Practice CER 0.06458 → 0.06435 (−0.023 points, interval includes 0).

Reading of the session: on this subject the deletion pattern is not a blank-bias offset. Every bias-level lever (blank, space, frequent rows) trades deletions for insertions and substitutions roughly one for one; the fine-tune's +0.113 on the blank is cosmetic. The remaining deletions sit in the recurrent stack's frame-level decisions, which no output-layer edit reaches.
