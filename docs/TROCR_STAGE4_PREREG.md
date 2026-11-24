# TrOCR stage 4: sessions on the Bern family, preregistration (written before any session, grid cell, sham, margin or held-out read of this stage)

Written 19.11.2026 (CET). Plan: `docs/TROCR_STAGE4_PLAN.md` (draft of 19.11.2026; its section numbers are used below). Epic: `docs/epics/EPIC_TROCR_TRACK.md`. Inputs: R15 (lineage), R16 (seed null), R17 (planted-defect control, Gate 3 met). Code: `wsp/stage4.py`, `wsp/landlock.py`, `scripts/trocr_stage4_*.py`, tests `tests/test_stage4.py`.

**Status of the choices.** Tobias gave the go-ahead to write this preregistration (relayed by the planning session, 19.11.2026) and on 20.11.2026 confirmed the seven decisions of plan section 10 as written here ("Defaults OK, go ahead"); every choice that rests on one of them is marked **[confirmed by Tobias, 20.11.2026]**. The same day he approved the exchange grid on practice (amendment 1) and the discovery-half reads of the top six cells per subject with arm I (amendment 2). **Not approved yet:** any read of the confirmation half, and the programmer sessions (P1, P2, P3, B), the shams and the margins.

## 1. Question

On a healthy fine-tuned model, does a programmer who edits weights find a held-out improvement that (i) a one-parameter interpolation toward the donor, (ii) a sham edit of the same size, and (iii) retraining noise do not already give? A negative answer is a result and is reported as one (plan section 0). Stage 3 (R17) settled that with the parent donor a repair is mostly a lookup of a parent tensor and that without it 1 of 8 planted defects was repaired; stage 4 therefore needs the donor-free arm to be a sandbox fact, not a rule in a brief (section 8).

## 2. Subjects and donors

| | subject | C0 held-out | donor (documented ancestor only) | donor C0 held-out |
|---|---|---:|---|---:|
| A | `dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2` (hub revision `ea9383f3c89d`) | 19.74 % | `dh-unibe/trocr-kurrent-XVI-XVII` (`a6bb6086ba3f`) | 19.83 % |
| B | `dh-unibe/trocr-kurrent` (`66e374e1e465`) | 26.41 % | `microsoft/trocr-base-handwritten` (`eaacaf452b06`) | not read here (an English IAM model; expected far worse) |

**[confirmed by Tobias, 20.11.2026: decision 1, both subjects, and the donor of B restricted to its documented ancestor.]** The ancestor of `trocr-kurrent` is `microsoft/trocr-base-handwritten` (`docs/MODELS_TROCR.md`, lineage tree; R15 puts the weights 0.228 apart, and the base lacks the ViT pooler, so the pooler is outside the editable set). That check is the planning session's assumption turned into a stated fact here; it is the documented one and R15 does not contradict it. Better relatives of B (`from-kurrent-v2` 19.74 %, `XVI-XVII` 19.83 %, 6.6 points better on the same lines) are **not donors**: an interpolation toward them improves B by construction, which is a merge with a better model. The interpolation toward the better of them is run once as a **ceiling comparator** (grid of section 6) and is reported separately, never as a programmer's or an arm-I gain. No third-party donor (`swe`, `stage1`: other decoder; `lamop`, `ru`: not family; R15). A consequence stated in advance: for B the donor is a model that cannot read the hand, so the with-donor and donor-free arms are expected to differ little on B.

## 3. Sets and settings

- **Practice:** `data/practice/trocr-blb-train-1200-seen.jsonl` (sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`; list without `seen_by` `09bdcb64503ed3aa3a41aa4d6d051990812494a85b8887c6c36c5a3ed64dee94`). Drawn with `python -m wsp.data build --seed 20261119 --n 1200` over the BLB **train side** (lassberg, korrespondenten, unbestimmt, manuskripte; never `*_eval`; 38,724 candidates), no model output in the draw. **1,194 lines** (6 of the 1,200 sampled crops were refused by the cutter: polygon outside the image), 759 pages, 216 documents. Independent of the stage-3 set (12 shared keys; both are samples of one pool); 28 degenerate lines (reference under 4 characters, taller than wide) are kept as in every earlier list. **[confirmed by Tobias, 20.11.2026: decision 3, 1,200 lines instead of 300.]** The 300-line set stays as the stage-3 comparison and is not used here.
- **Held-out:** `data/heldout/blb-eval-hand.jsonl` (2,272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`), read once per frozen model digest, outside the sandbox. Practice and held-out share **no document, no key and no exact line of 20 or more characters** (`data/practice/overlap-trocr-blb-train-1200.json`).
- **Object split (fixed now, before any read):** `data/heldout/blb-eval-hand.split.json` (sha256 `e6e5f6ccb62edac60100e4e4db64a564bd51671bf2e2bfc465a865b4d467d084`), made by `scripts/trocr_stage4_split.py` from the document ids and line counts only: objects in `random.Random(20261119)` shuffled order, each to the half with fewer lines. **Discovery** 17 objects, 1,147 lines; **confirmation** 12 objects, 1,125 lines; document-disjoint.
- **Out-of-domain control:** `bern-eval` (545 lines), read once for the finals that pass the held-out rule, after its own overlap audit against both subjects' and both donors' corpora; not part of the classes.
- **Generation (pinned, `TrocrSubject.GENERATION`, written in every `session.json`):** beam 4, 256 new tokens, no repeat of a trigram, length penalty 1.0, early stopping, no sampling, float32, batch 8.
- **Keep rule:** a pass since the edit, and the **page-clustered** paired 95 % interval of CER(after) − CER(last kept) entirely below **−0.005** (the seed null of R16: sd 0.14, range 0.33 over five seeds), 2,000 resamples, on the lines the donor did not train on. Budget 10 passes (the first is the baseline), 0 for the blind arm. Refusals are logged (section 8).
- **Hashes and digests** of every subject, donor, set and program are written into each run's `session.json` and `preregistration.md` stub by `wsp.run.new`.

## 4. Donors, `seen_by`, and what is assumed (amendment A3 **[confirmed by Tobias, 20.11.2026: decision 2]**)

The donor rule A3 applies in this track: the keep rule's interval is computed on the practice lines the **named donor did not train on** (`Session.unseen_by(donor)`); a donor-free session uses all lines. `seen_by` is `[]` on every practice line. What that rests on:

| model | corpus | audited against the 1,194 practice lines and the held-out lines |
|---|---|---|
| `from-kurrent-v2` (subject A), its twin `v2` and `full-v1` | the trainer jobs' 20 datasets, page XML kept | **yes** (`docs/trocr/audit_trained_overlap_1200.json`): no dataset hit, no exact line match; one `Laßberg` mention in a `full-v1` page (a printed 1875 page), no BLB document |
| `trocr-kurrent` (**subject B**) | 292,997 lines, "Kurrent dataset", many small collections | **no, not auditable here** |
| `trocr-kurrent-XVI-XVII` (donor of A) | about 1.58 M words | **no** (the question to the model authors, STAGE0 §8 item 4, is still open) |
| `microsoft/trocr-base-handwritten` (donor of B) | IAM, English | no (it cannot contain Kurrent lines of this hand; stated, not checked) |

So for subject B the practice set's independence from **the subject itself** is an assumption, not a result: if BLB lines are in its 293 k lines, its practice errors are training errors and any practice gain is suspect. The held-out classes do not depend on it (held-out is read after the fact against the subject's own held-out C0), but a null on B would be partly uninformative. This is stated as a limit, and the question to the authors is extended to `trocr-kurrent`.

## 5. Arms (per subject; each session is a fresh programmer reading only its pack)

| arm | who | donor | passes | note |
|---|---|---|---:|---|
| P1 | Sonnet 5.5 (`claude-sonnet-5-5`) | yes | 10 | |
| P2 | Fable 5.2 (`claude-fable-5-2`) | yes | 10 | |
| P3 | Sonnet 5.5 and Fable 5.2 (one each) | **absent from the sandbox** | 10 | Landlock allow-list, section 8. **[confirmed by Tobias, 20.11.2026: decision 5]** |
| B | Sonnet 5.5, blind | yes | 0 | `keep_unmeasured` only |
| R | replay: a second fresh session of the family with the better P1/P2 held-out result on the discovery half, blind to the first | yes | 10 | reproducibility |
| I | global interpolation toward the donor, one parameter | yes | 10 | grid of section 6, no programmer |
| S | five shams per kept edit | no | 0 | section 6 |
| M | each kept edit at 0.5 and 2 times its size | no | 0 | section 6 |

Programmers: **Sonnet 5.5 and Fable 5.2 only, no cross-vendor family [confirmed by Tobias, 20.11.2026: decision 4]**; the epic's "three programmers over at least two families" is met by arms P1, P2 and the blind B only as two model families plus a blind condition, which is stated as a deviation from the epic's wording unless decision 4 adds the cross-vendor family (the tool layer of `docs/XVENDOR_PLAN.md`). Per subject that is **6 programmer sessions** (P1, P2, P3 ×2, B, R), 12 for both (the plan's count of 7 and 14 included an extra P3 cell). Packs and the brief are built before the first session as in stage 3 (`scripts/make_trocr_packs.py` pattern), with the non-breaking-space check; the pack offers **magnitude traces only** (per-tensor distances, norms, the encoder/decoder move profile, the tied-embedding rows by movement decile) and **no direction readings** (R16: directions do not reproduce across seeds); a donor-free pack has no donor diff. Programs are written in full before the first pass and are not patched after.

## 6. Comparators and the grid

- **Sham (arm S, amendment A5 [confirmed by Tobias, 20.11.2026: decision 2]):** for each kept edit, five random changes of the **same tensors with the same Frobenius norm of the change** (`wsp.stage4.sham_deltas`, seed 20261119 and the sham index, i.i.d. Gaussian), read on the practice set and, for the edit chosen for confirmation, on the discovery half. A kept edit counts as larger than its shams only if its held-out gain exceeds the largest of the five.
- **Margins (arm M):** the kept edit at 0.5 and at 2 times its size along its own direction (`wsp.stage4.margin_states`), read on practice and discovery. An edit at a sharp optimum (both neighbours worse) is reported as such.
- **Arm I:** `(1−α)·subject + α·donor` over **all** shared tensors, α in {0.1, …, 0.9} (nine passes with the baseline = 10), the best grid point by practice CER (smaller α on a tie), accepted only if the harness's keep rule accepts it against the baseline. The ceiling comparator for B uses the same grid toward `from-kurrent-v2` and toward `XVI-XVII` and is a report, not an arm.
- **Exchange grid (plan section 5):** for each ordered (subject, donor) pair and its reverse, on the practice set: whole encoder, whole decoder, each of the 12 encoder and 12 decoder layers, the tied embedding with its norm, and the layer norms, moved at full and at half size (about 28 cells per pair and size). Shape and tokenizer identity are asserted first (R15). Cells are ranked on practice, the top six are read **once on the discovery half**, and the best of them **once on the confirmation half**. No programmer is involved. Readings fixed in advance: the same-recipe pair is a sanity case (R16 puts two runs of one recipe 0.108 apart) and an exchange across lineages is expected to fail.

## 7. Classes of a frozen model and Gate 4

Let Δ = CER(final) − CER(subject), in points, page-clustered paired 95 % interval, subject C0 read from the cached score of the unchanged digest. Every frozen final is read on the **discovery half**; within a subject only the best final by Δ on the discovery half (and each replay of it) is read on the confirmation half, once.

- **null:** nothing kept (digest unchanged).
- **harm:** Δ on the discovery half above +0.5 with the interval above 0.
- **practice only:** a gain on practice (kept edit), none on the discovery half (Δ above −0.5 or interval not below 0).
- **merge:** a discovery-half gain whose difference to the arm-I final, CER(final) − CER(I), has a page-clustered interval containing 0 (the interpolation already gives it).
- **gain:** Δ ≤ −0.5, interval below 0, not a merge, and, for the confirmed model, Δ ≤ −0.5 with interval below 0 on the confirmation half too.
- **Gate 4 (a finding):** at least one frozen model is a *gain* that also exceeds the largest of its five shams and arm I, and whose replay R has the same sign of Δ on the discovery half with edits in at least one common tensor group (same component and layer range). Anything else is the result of the stage, reported with the class of every session; practice-level gains that vanish held-out are the expected outcome of every earlier track.
- **Stops:** after this preregistration (cost and Tobias's go-ahead), after the grid (before the programmer sessions if the grid shows a large exchange gain they would only rediscover), and after Gate 4.

## 8. Harness check (done 19.11.2026, before this file's commit)

1. **`revert`/`keep` with no edit waiting** (the p01 crash of R17) is now a logged `refusal` event and not an exception; `keep` returns False, the model state is not touched (`tests/test_stage4.py`). Stage-3 programs are not re-run.
2. **File allow-list (Landlock, `wsp/landlock.py`)**: a session started with `restrict_files=True` runs under an irreversible allow-list written to `landlock.json`: its interpreter and venv, `wsp/`, its subject, the practice list and images, its own folder and a private temporary directory, the card-gate state file, `/dev` and `/proc` (CUDA needs them) and system libraries; **a donor path only for arms that have it**. No other model, run folder or held-out line can be opened. `unshare -Um` is refused on this host, so a mount namespace was not an option; Landlock needs no privilege. `scripts/trocr_stage4_noop.py {A|B} {donor|nodonor}` lists which of the 140 weight files and which held-out or other-run files the session can open and asserts: with the donor, the subject and the donor only; donor-free, the subject only; no held-out line, no other run. Result: see the record below.
3. **`wsp/stage4.py`:** sham generator, margin states, interpolation grid with its line search, object split, all pure functions with tests (`tests/test_stage4.py`).
4. **No-op sessions on both subjects** (C0 pass on the 1,194-line practice list on card 0, freeze, under the allow-list): see the record below. The repository tests are run and recorded there.

## 9. Cost and what is not decided

Passes on card 0 (rule: at least 12 GB free, no trainer job waiting, `wsp.card.ready`): the 300-line pass took 0.55 min; the 1,194-line pass is measured in the record below. About 12 programmer sessions at 10 passes (6 per subject, two of them 0-pass), arm I (10 passes each), shams (5 per kept edit, a few reads each), margins, the grid (about 56 cells per pair and size, practice then 6 discovery reads) and the held-out reads (about 14 finals on the discovery half, the best per subject on the confirmation half): the measured pass times (record below) replace the plan's estimate: 2.0 min on a free card, 12.2 min while another job shares it, so 9 to 10 h holds only on a free card and is about 5 times longer when shared. Programmer sessions cost tokens, roughly 12 sessions against the 32 of stage 3; not estimated. **Open, not this file's to decide:** decision 6 (stage 5, retrain and sibling average through the trainer, own OK after Gate 4); decision 7 (the question to the model authors about the parents' corpora and, now, `trocr-kurrent`'s).

## 10. Limits stated in advance

One held-out hand and one family; practice and held-out are the same hand, so a held-out gain is a gain on that hand; the with-donor results on subject A are lookups of parent tensors unless they beat arm I and the shams; subject B's practice independence is unaudited (section 4); the `seen_by` assumption for the parents' corpora; the discovery half has 17 objects and the confirmation half 12, so object-clustered intervals will be wide and the confirmation of a gain below 1 point is weak; the ceiling comparator for B is a merge by construction and says nothing about programmers.

## Record of the harness check

(Filled in after the runs; the text above is fixed.)

- Tests: `pytest tests` with the repo venv, 144 passed, 3 skipped.
- Listing no-ops (restricted sandbox, `scripts/trocr_stage4_noop.py`), subjects A and B, with and without donor: CHECK PASSED in all four. A donor-free session can open only its subject and its own run folder (1 weight file of the 139 found outside); no donor, no other run, no held-out file readable.
- C0 no-op with a pass on the 1,194-line practice set, restricted sandbox, donor-free: subject A CER 23.07 %, 731 s (card shared with the trainer's eval); subject B CER 28.22 %, 122 s (free card). Run folders `runs/20261119-trocr-stage4-noop-*`.

## Amendment 1 (20.11.2026): exchange grid run on practice only

The grid of section 6 ran on the practice set, 4 directions (A←XVI-XVII, B←IAM base, and the two reverses) × 28 cells × 2 sizes = 224 passes, 6.7 h on a free card 0, with the unedited recipient as baseline of each direction (results: `docs/trocr/stage4_grid.md`, `stage4_grid.jsonl`). Deviations from the text above: (1) the cluster bootstrap used 1,000 resamples, not the harness's 2,000; (2) the "norms" cell is every layer-norm tensor of encoder and decoder together (124 tensors), the "embed" cell is the tied token embedding with the decoder embedding norm; (3) the same-recipe sanity pair was not run (no sibling model of the same recipe is on this host); (4) the ranking and the reads of the top six cells on the discovery half were **not** done: the planning session approved the practice grid only, so those reads wait for a separate go-ahead.

## Amendment 2 (20.11.2026): ranking rule and discovery-half reads (written before the discovery reads; arm I on practice was already running)

Approved by Tobias on 20.11.2026 ("Yes, discovery reads"): the top six grid cells per subject and the best α of arm I are read once each on the discovery half (17 objects, 1,147 lines of `blb-eval-hand`); the confirmation half is not read. Rules fixed here, before any held-out model output exists:
1. **Cells considered:** only the subject's own direction (A←`trocr-kurrent-XVI-XVII`, B←`microsoft/trocr-base-handwritten`); the reverse directions are not read held-out. Both sizes (1.0 and 0.5) are ranked together.
2. **Ranking:** by the practice point estimate of CER(cell) − CER(baseline), lowest first, the smaller factor on a tie; the first six per subject, whether or not they overlap (a layer cell inside a whole-decoder cell counts as its own). No interval or minimum effect is required to be read; the discovery class applies afterwards.
3. **Arm I:** α in {0.1, …, 0.9} on the practice set, all shared tensors except the unused ViT pooler (the IAM base's pooler is random-initialised); the best α by practice CER, the smaller α on a tie, read on the discovery half whether or not the keep rule accepts it on practice. The ceiling comparators for B (toward `from-kurrent-v2` and `XVI-XVII`) are **not** run in this step.
4. **Baseline:** each subject's unedited model is read afresh on the discovery half in the same script (batch 8, same generation settings), so every Δ is paired with a baseline of the same batch composition. Clustered 95 % interval by page, 2,000 resamples (amendment 1's 1,000 applied to the grid only).
5. Reported: the ranking with practice Δ, the discovery Δ with its interval for the six cells and for arm I, and whether a cell's discovery interval overlaps arm I's. Scripts: `scripts/trocr_stage4_discovery.py`; results `docs/trocr/stage4_armI.jsonl`, `stage4_discovery.jsonl`.

## Amendment 3 (20.11.2026): subject A sessions, harness defect and reruns

Approved by Tobias on 20.11.2026 ("Yes, A first"): programmer sessions on subject A only (P1, P2, P3 ×2, B), practice set only, frozen models read on the discovery half only; no confirmation-half read. Packs and brief: `runs/trocr-stage4/A/` and `runs/trocr-stage4/PROGRAMMER_BRIEF.md` (committed before any programmer started; magnitude traces only, no cosine). Programs were written by fresh subagents (`sonnet` for P1, P3s, B; `fable` for P2, P3f) and are not patched.

**Harness defect and reruns.** The first runs of P3s and P1 crashed at the first pass, and P2 and P3f crashed after C0 and their first edit, all with `PermissionError` on `~/atr-cache/wsp/card-state.json`: the card gate reads the watcher's state file, the watcher replaces that file, and a Landlock rule keeps the old inode, so the new file was denied. No programmer error, no refusal, no edit was ever measured or kept; the four run folders (`runs/20261120-trocr-stage4-A-{p3s,p1,p2,p3f}-b10-1`) stay in the repository as records and their results are **discarded**. Fix: the allow-list no longer contains the state file; the card gate runs in the runner before a session starts (`WSP_CARD_GATE=outside` in the sandbox environment, `scripts/run_trocr_stage4_session.py`). The **same unchanged programs** were rerun (`-b10-2`). A mid-session start of a trainer job on card 0 is therefore no longer caught inside a session (the runner checks before it starts only). The B session (budget 0, no pass) ran once, before the fix, and was not affected. Information that carried over to the reruns: only that C0 can be read under the sandbox.

## Amendment 4 (21.11.2026): shams, margins and the replay on subject A

(1) The replay R was a fresh Sonnet 5.5 session (the family of the better of P1 and P2 on the discovery half, P1), blind to the first, with the donor; its program is unpatched. (2) Shams and margins were computed on the **net change** of a session (final state minus the subject), not edit by edit, and **every** sham and margin variant was read on the discovery half as well as on practice (the text above reads only the chosen edit on discovery); discovery reads only, no confirmation half. Seeds: 20261119 and the sham index. (3) **Gap in section 7:** *merge* is defined as "the difference to arm I has an interval containing 0". P1, P2 and R are discovery gains that are worse than arm I with the difference interval above 0, so by the letter they are neither a merge nor does any of them meet Gate 4 (they do not exceed arm I); they are reported as "below arm I", every kept edit being a donor pull. Whether this counts as *merge* is left to Tobias; no confirmation read was made. (4) The sessions file `stage4_A_sessions.jsonl` also holds rows from the discarded first runs of amendment 3; only rows of the `-b10-2` runs, `r` and `b` count. Results: `docs/trocr/stage4_A_results.md`.
