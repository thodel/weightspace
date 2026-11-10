# Fact-check of `paper/manuscript.md`, draft 3 (main at 1a23b55)

Checked 10 November 2026 (CET) by a second Claude instance on h1, read-only. Nothing was edited, committed or pushed; this is the only file written.

**Scope.** Abstract, Sections 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 8.2, 9.3, RQ1–RQ3 in 8.4, the conclusion (11), plus the open line-count item.

**Method.** Every number or did/did-not claim was traced to a committed source: `docs/RESULTS.md` (R-numbers), `docs/CONTROLS_REPORT.md`, `docs/C4_RETRAIN.md`, `docs/ADAPTER_REPORT.md`, `docs/MODEL_FAMILY.md`, `docs/ZOO_SURVEY.md`, `docs/PROTOCOL.md`, `docs/controls/*.json|md`, `docs/c4/*`, `docs/family/*`, `docs/baselines/*` and `runs/*/summary.json` / `log.jsonl`. The Fable 5.1 runs are not on `main`, so I read them from `origin/claude/article-controls-8803eb` with `git show`, with no checkout. I did not recompute anything from model weights or per-line readings; "matches" means the manuscript agrees with the committed record, not that I re-measured it. Where two committed sources disagree, I say so.

**Verdicts.** `matches`, `MISMATCH` (both values given), `unsourced` (no committed source), `overstated` (number right, wording stronger than the evidence), `partly`.

---

## Headline findings

Ranked list with detail at the end (§14). The five that matter most before the text leaves the repo:

1. **Table 10 (7.7) carries page-level intervals that no source supports.** Row 2's `[−0.72, −0.06]` is the figure `CONTROLS_REPORT.md` withdrew. Row 3's `[−0.67, +0.04]` is in no committed file. The correct two-stage values are `[−0.84, +0.04]` and `[−0.84, +0.19]`. As printed, row 2 excludes zero, against 7.8, RQ2 and the abstract.
2. **RQ2's last sentence says "a retraining on corrected data was not run".** Section 7.11 and the sentence before it say it was.
3. **The latin_incunabula candidate v3 (RESULTS R12) is nowhere in the manuscript.** It qualifies under a later rule with minimum effects, whose thresholds were taken from R9's numbers. The abstract, 7.10, RQ2 and the conclusion say that neither candidate qualifies and that the rule was not changed.
4. **"None of the steps excludes zero" (7.7, 7.8, RQ2) is contradicted by the table it sits next to.** The full-step row's page interval is `[+0.05, +2.19]`. Also, "clearly worse from 0.75 on" overstates α = 0.75, whose intervals contain 0.
5. **The Fable 5.1 numbers (7.7, RQ2, 9.2) are not on `main`.** Only the one-line summary in `RESULTS.md` R7 is there, while `CONTROLS_REPORT.md` on `main` still says no Fable attempt exists.

**The 5,243 vs 5,441 item is settled** (§12): 5,441 is right. 5,243 is the sum of two quantities that are not like-for-like.

---

## 1. Abstract

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| 43 kraken models served, compared as networks | 43 | `MODEL_FAMILY.md` L12: 43 = 37 Zenodo + 6 in-house | matches. `ZOO_SURVEY.md` read only 30 of 37 (seven refused), and `MODEL_FAMILY.md` says weights for all 43 were obtained; see §14 item 19. |
| "six in-house models and their public parent" | six + parent | Table 1: 6 models. Only family 1 has a public parent (`cursive_inzigkofen`); `german-xix-v1`'s base is `bohemian_19th`, also public (MODEL_FAMILY). | partly: "their public parent" is singular but there are two lineages |
| Null of five seeds | 5 | R6 | matches |
| Blank-bias rise survives the seed null; "size of a weight change alone" does not | stated | R6 | matches |
| 15 edits proposed on training lines, none improved held-out CER | 15 | R2: 7 edits, R3: 8 edits | matches in count. Overstated: held-out was scored once per session on the final model (PROTOCOL §3), so the claim is about two final models, not 15 individual edits. Session 1's −0.010 is nominally a gain, with interval `[−0.048, +0.026]`. |
| Practice lines not trained on: one family gave nothing, the other a small gain that is a plain interpolation, "no page-level interval excluding zero" | stated | CONTROLS §3.3, §6, §9c | matches the sources. **Contradicted by manuscript Table 10 row 2** (§3 below). |
| Positive control: repairs large localised defects; list does as well as programmer; only a diffuse defect repaired by the programmer alone | stated | CONTROLS §2.3, §9b | matches |
| Retraining lowered held-out CER by 0.55 (protocol lr) and 0.99 (larger lr) | 0.55 / 0.99 | C4 T3: −0.55 (sd 0.01), −0.99 (sd 0.02) | matches |
| "at least as much as the one edit that survived" | stated | edit −0.39; arm A −0.54..−0.56 | matches in size. "Survived" is generous: the edit's page-clustered interval is `[−0.84, +0.04]`. It survived the sham control, not the page-level interval. |
| Exchange gave "two candidates", Fraktur CER down 0.2 to 2.0 points | 0.2–2.0 | R9: v1 −2.04 / −0.44; v2 −1.08 / −0.19 | matches |
| "by a pre-registered rule neither qualifies" | stated | R9 | matches R9. **Omits v3 (R12), which qualifies under a later pre-registered rule.** §14 item 3. |
| Data defects: 44 of 50 training pages with wrong image | 44 / 50 | R3, MODELS.md, C4_RETRAIN A.1 | matches |
| Handwriting held-out set "of which half was a printed book" | half | PHASE4 L27: 2,825 of 5,628 lines (50.2 %) | matches |
| Adapter: one kept edit acted on output length, not reading; metric too unstable | stated | ADAPTER_REPORT A4 | matches |

---

## 2. Section 7.6, positive control

Source: `CONTROLS_REPORT.md` §2, §9, §9b; `docs/controls/planted_table.md`; R7.

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| Eight corruptions; held-out 2,272 lines; uncorrupted 9.55 % | 8 / 2,272 / 9.55 | §2 | matches |
| Outcome classes: repaired ≥ 80 % and interval below 0; partly 20–80; not repaired < 20 | stated | §2 preregistered classes | matches |
| Damage column (+10.10, +60.69, +18.20, +1.30, −0.03, +11.62, +0.49, +90.45) | as listed | §2.1 | matches (all 8) |
| Table 9 cells (a)–(d), CER in brackets | as listed | §2.2 and `planted_table.md` | matches (all 32 cells checked). k5 (b) "halved, no effect" matches "×0.5 applied: 9.55". |
| "six corruptions that cost at least one point (1.3 to 90)" | 6 | §2.3 | matches |
| (a), (b), (c) within 0.22 points in "each of the 18 cases" | 18 / 0.22 | §2.2: 19 repaired cells including k4 (d); (a)(b)(c) × 6 = 18; max +0.22 (k8) | matches |
| (d) repaired only the blank offset and partly three others | stated | k1, k2, k6 partly; k3, k8 not | matches |
| "harness accepted the right edit and refused the wrong ones" | stated | §2.3 (i) | matches |
| Defects "show at once" in the diff (cosine −0.99, zeros, half norm); repairs borrow from the parent | stated | §2.3 "does not show" (i), (ii) | matches |
| "programmer adds nothing over a list of rules … forward passes add nothing over readings" | stated | §2.3 (ii) | matches |
| "line-level intervals" (text) and "Intervals resample lines" (Table 9 caption) | line level | CONTROLS §9 and `planted_table.md`: every cell now carries a page-clustered interval and "classes are decided on the page interval"; no class changes | MISMATCH (low): the sources decided classes on page intervals. Manuscript says line-level. |
| Diffuse set: k9 noise 10 % on every tensor, +54 points; (b) wrote "no localised defect"; (c) no rule fired; (a) pulled non-output tensors a quarter toward the parent, 63.8 → 18.5, 83 % | as stated | §9b | matches |
| "the one cell of the study in which forward passes bought something reading did not" | stated | §9b (i) | matches |
| k10 block noise 60 %: 51–70 % partly | 51–70 % | §9b: (a) 66, (b) 70, (c) 51, (d) 0 | matches |
| Table 11 (cited in text as "Table 11") | — | numbering: the diffuse table is Table **10** | MISMATCH (cross-reference). Tables count 1–13 in order; planted = 9, planted2 = 10, validation = 11, sham = 12. |
| k11 "all LSTM × 0.9 … too little damage to count (+0.31)" and table cells "null by the 0.5-point rule / null / null / null" | null | §9b: (a) 9.55 (100 %), (b) 100 %, (c) 102 %, (d) 74 %. The class is "null corruption" but the arms recovered it. | MISMATCH (low-medium): the table caption says "share of the damage recovered", yet the k11 row gives the class, not the share. Source says three arms repaired it. |
| Unrelated donor: zero row `e` (c) 19 %, (a)(b) no edit; zero `C_3` (a) 1 %; halved block undone without donor | 19 / 1 / 100 % | §9b | matches. Table cells k6-u (c) 42 % ✓. |
| "Sonnet programmers", one attempt per cell | stated | §1, §9b | matches |

---

## 3. Section 7.7, practice sets that are not training data

Sources: `CONTROLS_REPORT.md` §3, §9, §9c (branch version for Fable); R7; `runs/20261107-val-*`; `docs/controls/cluster_intervals.md`.

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| Kurrent: 1,651 training pages all trained on; validation = the 157 eval pages; best checkpoint by validation accuracy | stated | §3.1 (`best_0.9127`) | matches |
| Seven objects, 300 lines; held-out left with 1,718 lines, unedited 8.41 % | 7 / 300 / 1,718 / 8.41 | §3.2 | matches |
| 13.83 % unseen vs 6.46 % on training lines | stated | §3.2 | matches |
| Bern: validation 17 of 168 pages; all six Hertwig validation pages are training pages of its parent and of the donor `hertwig-v2` | stated | §3.1 | matches. "its parent and … the donor `hertwig-v2`" is one model counted twice (the parent is `hertwig-v2`); cosmetic. |
| Table 10 row 1: `xix-blb-v1`, "3 programmers and a replay", **0 of 28 edits**, 8.41 → 8.41 | 28 | CONTROLS §3.3 text: "0 of 28 edits kept". The logs give a1 7 + a2 7 + a3 5 + replay 8 = **27**. | MISMATCH (low): the manuscript copies "28" from the report, but run logs and the report's own table sum to 27. (Opus 4 + Haiku 4 add 8 more.) |
| Row 2: `combined-v1`, 3 programmers, 2 of 8, 19.33 → 18.94, −0.39 `[−0.69, −0.08]` | stated | §3.3 a1; summaries: a1 8 edits/2 kept, a2 7/2, a3 8/2 | matches |
| Row 2: "**by page `[−0.72, −0.06]`**" | `[−0.72, −0.06]` | §9: `[−0.838, +0.035]`. CONTROLS L5: the `[−0.72, −0.06]` figure "came from a one-stage page resample and is **withdrawn**". | **MISMATCH (high).** Printed value excludes 0; correct value contains it. Contradicts Table 12 (`[−0.84, +0.04]`) three paragraphs later, and 7.8, RQ2, the abstract and the conclusion. |
| Row 3: replay, 1 of 8, 19.33 → 19.01, −0.32 `[−0.70, +0.07]`, "**by page `[−0.67, +0.04]`**" | `[−0.67, +0.04]` | §9 / `cluster_intervals.md`: `[−0.840, +0.189]`. `[−0.67, +0.04]` appears in no committed file. | **MISMATCH (high) / unsourced.** Probably also a one-stage figure. |
| Kept edits: blank −0.3 and non-output α = 0.25 toward the sibling | stated | §3.3 | matches |
| Reads 18.94 vs 19.33; text cites "Table 10" | Table 10 | validation table is Table **11** | MISMATCH (cross-reference) |
| Gain −0.49 on Hertwig lines `[−1.10, +0.04]`; −0.30 on Schlussredaktion `[−0.58, −0.01]` | stated | §3.3 | matches |
| "first held-out gain of the project and small"; "a merge, not a repair"; no comparison with siblings average / sham of the same size | stated | §3.3 Reading (i), later §6 | matches. The text says "it has not been compared with an average of siblings or a per-block sweep, nor with a sham"; §6 did exactly those. See §4 below (7.8 supplies them). |
| Donor rule: output-layer transplant refused, judged on 301 lines, interval `[+0.32, +0.83]` | stated | §3.3 | matches |
| Replay's one kept edit is the first LSTM layer from the sibling, does not clearly transfer | `L_12` | §3.3 (`L_12` = first LSTM) | matches |
| Further programmers: Opus 5.5, Haiku 5.5, Fable 5.1 (run 9 Nov, earlier attempt failed for lack of credits) | stated | branch `article-controls-8803eb`, commit `1c27ec1` (9 Nov, 08:12 CET): `f1` runs; `main` `CONTROLS_REPORT.md` L143: "no Fable attempt exists" | partly: sourced, but the run folders (`runs/20261109-val-*-f1-*`) and the updated report exist only on an unmerged branch. On `main` only `RESULTS.md` R7 (one sentence) records them. |
| Kurrent: none kept an edit → "0 kept in seven attempts" | 7 | 4 (a1–a3, replay) + Opus + Haiku + Fable; branch: Fable 10 passes / 9 edits / 0 kept; R7 "0 in 7" | matches |
| Bern: Opus −0.19, Haiku −0.31, Fable −0.02 | stated | §9c: −0.193, −0.314; Fable −0.018 (branch `f1`: 19.33 → 19.31) | matches |
| Fable "half-way in a single step": practice 13.86 → 11.60, held-out did not move | stated | branch: last-pass 0.1160; kept α = 0.5 toward the sibling | matches |
| "exactly the half-step row of Table 12" | exactly | half-step + blank: practice 11.54, −0.04; no-blank α = 0.5: practice 11.66, +0.02; Fable 11.60, −0.018 | overstated: "close to", not exactly |
| "None of the page-clustered intervals excludes zero (Section 7.8)" | stated | for gains: true per §9c (Fable `[−0.623, +0.603]`, Opus `[−0.788, +0.369]`, Haiku `[−0.771, +0.086]`) | matches for gains, but **contradicted by Table 10 row 2 as printed** |

---

## 4. Section 7.8, repair, merge or noise?

Sources: `CONTROLS_REPORT.md` §6, §9; `docs/controls/combined_controls.json`.

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| 20 shams, same Frobenius norm per non-output tensor + same blank shift; mean −0.013, sd 0.062, range −0.199..+0.097; 0 of 20 reach −0.39; rule kept 0 | stated | §6(a), R8(a) | matches |
| Blank alone +0.01; quarter-step alone −0.31; sum −0.39 | stated | §6 table | matches |
| "held-out improves for steps 0.1 to 0.25, zero at 0.4 to 0.5, including the uniform average (−0.02)" | stated | §6 | matches (point estimates; with-blank α = 0.4 is −0.16) |
| "**clearly worse from 0.75 on**" | clearly | α = 0.75: +0.43 `[−0.05, +0.89]`, pages `[−0.36, +1.29]`; with blank +0.29 `[−0.21, +0.75]`. Source says "the held-out loss is **clear by α = 1**". | **overstated.** At 0.75 both intervals contain 0. |
| Practice CER keeps falling to 0.75, where held-out is 0.3–0.4 worse | stated | §6 (4) | matches |
| "the keep rule refused the larger steps only because it is conservative" | stated | §6 (4) | matches |
| "**with page-resampled intervals, none of the steps excludes zero**" | none | `combined_controls.json`: α = 1.0, pages `[+0.048, +2.193]`; Table 12 row "full step" prints `[+0.05, +2.19]` | **MISMATCH (medium).** The statement is true for gains only. The full step's loss does exclude 0 (barely). The source report (§6 (3)) has the same slip. Repeated in 7.7 and RQ2. |
| Table 12 cells (practice, held-out, lines, pages) for all rows | as listed | §6 table / JSON | matches (all 7 rows; sham row practice 13.82 ✓) |
| Page `[−0.04, +1.08]` for the Schlussredaktion "loss" | `[−0.04, +1.08]` | §9: `[−0.043, +1.076]` | matches |
| "**Three** earlier statements need correction" | three | only two are listed (practice gain, Schlussredaktion loss); §10 says "two earlier line-level findings" | MISMATCH (low): count. The third may be the withdrawn Table 10 interval, which is still in the text. |

---

## 5. Section 7.9, additivity

Source: `CONTROLS_REPORT.md` §7; `docs/controls/global_interp.json`.

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| Eight blocks moved jointly, α quarter steps 0 to 1.5; 300 practice lines, 1,718 held-out | stated | §7 | matches |
| Extrapolation costs: +0.29 at 1.25, +2.05 at 1.5 (held-out) | stated | §7 table | matches |
| Joint − sum on held-out: +3.81 at the parent (pages `[+2.57, +4.98]`), +2.23, +2.19, +0.38 | stated | §7 table | matches |
| "on the 300 practice lines the sign is the same but intervals contain zero" | stated | practice joint−sum: +2.48, +1.63, +2.07, +0.46, −0.10 (1.25), +0.16 | partly: sign holds at α ≤ 0.75 and 1.5, not at 1.25 (−0.10). Held-out at α = 1.5 is +1.03 `[+0.51, +1.59]`, significant, and not mentioned. |
| Pairs at α = 0.5: correlation 0.98 / 0.97, mean interaction +0.11 / +0.07 | stated | §7 (iii) | matches |
| 9.1 item 2 "+0.4 to +3.8 points" | stated | §7 (ii) | matches |

---

## 6. Section 7.10, block exchange

Sources: R4, R9; `docs/family/candidates-latin_incunabula.json`; `MODEL_FAMILY.md`; `runs/20261107-latin_incunabula-exchange-{1,2}/preregistration.md`.

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| "grid of 240 readings" | 240 | MODEL_FAMILY L178: "240 readings" | matches |
| First grid: no block's fine-tune undone free, first convolution closest; recurrent trade one for one or worse; output layer no lever; stabilisation only in conv blocks toward a better general model | stated | MODEL_FAMILY L186–207 | matches. Nuance: the same file says `C_0` half-way costs nothing (−0.02). "No block … at no cost" holds for the full transplant only. |
| Candidate v1 construction (`Gn_1`, `Gn_3` from `catmus_print_large`, `O_15` half-way); chosen on practice, confirmed on disjoint lines, scored once | stated | R4 | matches |
| Held-out: 4,374 lines "of **60** pages" | 60 pages | R4 and R9 prereg: 60 pages drawn, **4,374 lines on 58 of them** | MISMATCH (low) |
| Audit: no shared page, line or crop | stated | R4 | matches |
| Own collection 4.99 vs 5.01, −0.03 `[−0.07, +0.02]` | stated | R4 | matches |
| Serving path 58 pages: better on 40, worse 17, equal 1; median page CER 4.05 vs 4.16; interval of median difference `[−0.27, −0.10]` | stated | R4 | matches |
| Fraktur: NewsEye 2,706 lines 28.56 → 26.52 (−2.04 `[−2.22, −1.86]`); GT-Fraktur 1,311 lines 20.13 → 19.70 (−0.44 `[−0.57, −0.31]`) | stated | R9 | matches |
| MUFI 532 → 731 (+0.11, interval excludes 0); lowercase −0.08 | stated | R4/R9; JSON: lowercase share −0.0008 (= −0.08 points) | matches |
| v2: MUFI 532 → 551 (+0.01), incunabula +0.04, Fraktur −1.08 / −0.19 | stated | R9 | matches. R12 notes v2's "+0.04 worse" interval is `[−0.01, +0.10]` page-clustered and would not hold. The manuscript uses only the line-level interval. |
| Rule: qualifies if no abbreviation loss, own collection not worse, neither Fraktur set worse; neither qualifies; v1 registered and disabled, v2 not registered | stated | R9; prereg commit `0c8f4ff` (7 Nov, 20:49 CET) | matches |
| "a rule written and committed before these scores" | stated | The preregistration itself says: "The only held-out numbers known when this was written are v1's on `reichenau-heldout` (R4)", i.e. v1's own-collection score (−0.03) and its MUFI loss (532 → 731) were known. | **overstated.** True for v2 and for both Fraktur sets. Not true for v1's Reichenau result, which the abbreviation criterion was written against. |
| "the rule set no minimum size, and it was not changed afterwards" | stated | R9: true. R12 (8 Nov, 19:20 CET prereg `fdf25e2`): a **new** rule with minimum effects ("chosen from R9's numbers: v1 MUFI +0.110, v2 +0.010; v2 Reichenau +0.04") under which candidate **v3 qualifies**. | **MISMATCH by omission (high).** R9's rule was not changed, but a successor rule fitted to R9's numbers produced a qualifying candidate, and none of it is in the manuscript. Also: the three Latin held-out sets have now been read by three candidates (R12 closing note). |
| "A use case that wants Fraktur robustness would choose the first" | stated | R9 final paragraph | matches |

---

## 7. Section 7.11, the gradient alternative (C4)

Sources: `docs/C4_RETRAIN.md` part B, `docs/c4/{runs,results,tables}`; preregistration commit `2ea91a3` (8 Nov, 09:39 CET) precedes results commit `2833b24` (10:16 CET).

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| Subject `combined-v1`; training side of its job 151 pages, 4,927 lines; Hertwig from the 22 Oct export; validation 17 pages; held-out 545 lines / 14 pages, read once per model | stated | A.2, B.1 | matches |
| Arm A 5 epochs lr 1e-5 three seeds; arm B 5e-5 secondary | stated | A.3 | matches |
| batch 16, augmentation, NFD, kraken 7.0.2, CPU scoring, nothing registered | stated | A.3, B.1 | matches |
| "runs took **3 to 5** minutes" | 3–5 | B.1: "2 to 5 minutes"; `runs.json`: 161 s to 287 s | MISMATCH (trivial): 2 min 41 s |
| Reading rule: detectable ≤ −0.20 in all seeds; equal to interpolation if −0.57..−0.21 | stated | A.4 | matches |
| Arm A 19.33 → 18.77–18.79; −0.54, −0.56, −0.56; page intervals about `[−1.1, −0.1]` each excluding 0 | stated | T2: `[−1.06, −0.09]`, `[−1.09, −0.09]`, `[−1.06, −0.10]` | matches |
| Interpolation edit −0.39, page interval includes 0 | stated | T2 | matches |
| Arm A vs interp difference −0.15..−0.18, inside intervals | stated | T2 | matches |
| Arm B −0.99 mean (−0.98..−1.02), page interval about `[−1.7, −0.3]`; larger than interp by 0.6, page interval "reaches zero" | stated | T2/T3: −0.59/−0.63/−0.60 vs interp; pages `[−1.32, +0.04]`, `[−1.34, −0.01]`, `[−1.33, +0.09]` | matches. Seed 2's page interval just excludes 0 (−0.01). |
| "seeds differ by 0.01 to 0.02 points within an arm" | 0.01–0.02 | those are the **standard deviations** (T3). The range within arm B is 0.04 (−0.98 to −1.02). | partly (low) |
| Much less than the seed null's 0.18, explained by training from a small step | stated | B.2 | matches |
| Gain on Hertwig lines −1.30 (A), −2.22 (B), 287 lines; Schlussredaktion +0.04, −0.01 (258 lines) | stated | T3 | matches |
| Collapsed lines 34 → 24–28 | stated | T2 (A 26–28, B 24–27) | matches |
| No run on uncorrected pairs or other schedule; validation accuracy still rising at epoch 5; 14 pages, six Hertwig | stated | B.3, T1 | matches |
| "a negative result for editing weights as a repair method on this subject" | stated | B.3 (1) | matches the source wording |
| "does at least what the one weight edit does … and does more at a larger step" | stated | B.2 | partly: arm A is "equal" by rule. Arm B's advantage over interp has page interval touching 0 and arm B is the secondary arm at a 5× larger learning rate. See §14 item 6. |

---

## 8. Section 8.2, why output-layer edits did not do what the readings suggested

Sources: `CONTROLS_REPORT.md` §8; R8(c); `docs/controls/margins.json`; BLOCK_STATS L44.

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| 696 deleted spaces, `e`, `n` in 418 lines (of 1,718) | stated | §8 | matches |
| 127 (18 %) have no frame between the neighbours; other 569 | stated | §8 | matches |
| Median 3.2 nats below blank, factor 24; 63 % more than two nats away | 3.2 / 24 / 63 % | §8: 3.17; e^3.17 ≈ 23.8; share M1 < 2 = 37 % | matches |
| Blank bias lowered by 0.1, 0.5, 1.0 recovers about 1 in 40, 1 in 9, 1 in 5 | stated | §8: session-1 state is −0.113; predicted 2.6 / 9.1 / 19.0 %; measured 2.7 / 11.9 / 21.8 % | matches (rounding of −0.113 to 0.1) |
| At 1.0, 152 recovered deletions vs 133 new insertions | 152 / 133 | §8 | matches |
| "predicted and measured recoveries agree in size" | stated | §8 | matches |
| Single recurrent blocks carry the largest effects at α = 0.5; halving the last LSTM block loses word boundaries | stated | §9 text (+1.2 to +2.3); BLOCK_STATS L44 (`L_16`, +1.37 on Space) | matches |

---

## 9. Section 9.3, adapter probe

Sources: `ADAPTER_REPORT.md`, R10, `runs/20261107-adapter-v2-b10-1/summary.json`, `docs/baselines/a4-heldout134-result.json`.

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| Rank 64, scale 2, 128 modules, 340 MB, Qwen3.5-4B, Kurrent pages | stated | A0.1 | matches |
| 49 of 733 pages at the 3,072-token ceiling; CER 0.418 vs 0.250 clipped | stated | A0.1 | matches |
| Keep rule and protocol "those of Section 7, with pages in place of lines" | stated | A4 prereg: "interval below 0 **and pages at the cap not up**" | partly (low): an extra condition is undisclosed |
| Norms tight across three healthy adapters, 0.80–0.93 of siblings' median | stated | A1 | matches |
| "energy sits in the gate and up projections of the **late** layers" | stated | A1: gate 47 %, up 28 %; top-energy layers 27, 23, **0**, 31, **3** | partly: not only late layers |
| Logit lens names end-of-turn directions, mostly noise tokens | stated | A1 | matches |
| Cosine 0.10; no shared A matrix in any of 128 modules; no donor | stated | A1 | matches |
| Of six planted defects, two did damage beyond noise; doubled scale CER 3.76 (64 pages) vs 0.51 clean, repaired by all three arms to 0.517 | stated | A3 | matches. Not disclosed: arms (a)(b) were run for this defect only, and their pack printed the doubled `lora_alpha` next to the siblings' 128 ("checks the machinery, not insight", A3). |
| Eight late modules ×3 (+24): rule list recognised, repair refused by the keep rule (32 practice pages) | stated | A3 p6 | matches |
| Four other defects did no damage | stated | A3 | matches |
| One real session: six edits within seven of ten passes, one kept: `q_proj` ×0.8 of one attention layer | stated | A4; `summary.json`: used 7, edits 6, kept 1 | matches |
| Held-out 134 pages 0.685 → 0.569; page interval `[−0.271, −0.001]`; sham +0.032 `[−0.060, +0.101]` | stated | `a4-heldout134-result.json`: Δ −0.115 `[−0.271, −0.001]`; sham A4 | matches |
| Clipped CER 0.354 → 0.351; length ratio 1.33 → 1.22; 4 of 22 capped pages stop | stated | A4 (1.215) | matches |
| Engines give identical text on 10 of 61 pages; merged vs adapter differs on seven of ten; any change reshuffles about two thirds of pages | stated | A0.3, A4 (40 of 134 identical; 12 of 32 for the numerical sham) | matches |
| Base alone loops more: 12 of 32 at the ceiling vs 8 | stated | A4 controls | matches |
| "not distinguishable from what chance gives" | stated | A4 | matches the report. Note the frozen-vs-sham difference is `[−0.300, −0.041]`; the report's argument is that one sham is not a null distribution. |
| "a damaging and visible defect is repaired and a localised invisible one is not" (also abstract, conclusion) | stated | A3: p6 is not invisible (the rule list identified all eight modules) and not localised (8 modules); its repair was **refused by the keep rule**. The four single-module defects did no damage, so there was nothing to repair. | **overstated / mischaracterised (medium).** |

---

## 10. Section 8.4, RQ1–RQ3

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| RQ1: seed null of five runs separates what is specific to fine-tuning (blank bias relative, rare-row growth) from what is not (conv depth pattern, "rare rows move most") | stated | R6 | matches |
| RQ1: lineage recoverable across 43 models for fine-tunes that did not move far | 43 | MODEL_FAMILY (43, cosine); distance 0.1–0.3 vs 1.41 and "11 of 13" are from `ZOO_SURVEY.md` (30 models) | partly: the figures are from the 30-model survey. See §14 item 19. |
| RQ2: seven edits in Session 1 (two kept), eight in Session 2 (one kept); held-out −0.010 and +0.15, intervals contain zero | stated | R2, R3 | matches |
| RQ2: seven attempts by four models; Kurrent 0 kept in seven; Bern −0.02 to −0.39 | stated | §3 above | matches |
| RQ2: "no page-clustered interval excludes zero (Section 7.8)" | stated | see §4 | overstated: full step excludes 0; Table 10 row 2 as printed excludes 0 |
| RQ2: positive-control summary | stated | §2 | matches |
| RQ2: "so for repairing this model **gradient training is the better tool**" | stated | C4 B.2/B.3 | **overstated (medium).** Arm A is "equal" by the preregistered rule. Arm B is a secondary arm, with a page interval vs interp reaching 0. There is no control on uncorrected pairs, and B.3 says additional training might also help. |
| RQ2: "the only candidates that do more than interpolate … neither qualifies" | stated | R9; R12 | MISMATCH by omission (v3), §6 |
| RQ2 last sentence: "The sessions remain non-interactive and have one or few runs per cell; **a retraining on corrected data was not run.**" | not run | 7.11, abstract, conclusion and the sentence before in RQ2 | **MISMATCH (high):** internal contradiction. Likely a leftover from draft 2. §10 correctly says the retraining on *uncorrected* pairs was not run. |
| RQ3: mechanism right, signs and sizes wrong; defect not found by readings; baseline measurement made it visible | stated | §8.1, appendix A.1 | matches |
| RQ3: "none of that kind was present in the real models" | stated | CONTROLS §2.3: absence of a found defect "carries information" about localised visible defects | overstated (low): an inference, not a measurement |

---

## 11. Section 11, conclusion

| Claim | Manuscript | Source | Verdict |
|---|---|---|---|
| 43 kraken models; lineage; generic front end; two traces survive the seed null | stated | R5, R6, MODEL_FAMILY | matches. The seed-null results come from five pairs of in-house models, not from 43. |
| "Fifteen edits proposed in two pre-specified sessions improved no model on held-out data" | stated | R2, R3 | matches (see §1 on final models vs edits). Session 1 ran under the earlier keep rule (PROTOCOL §3). |
| Positive control summary | stated | §2 | matches |
| "one small held-out gain, an interpolation … that random edits do not match, but page-clustered intervals do not separate from zero" | stated | §6, §9 | matches the sources; contradicted by Table 10 row 2 as printed |
| "A systematic exchange of blocks gave two candidates … neither qualifies" | stated | R9 | MISMATCH by omission (v3) |
| "A retraining … did at least as well as the best edit and better at a larger step" | stated | C4 | matches in point estimates; overstated as in §7 |
| "found by ordinary quality assurance" | stated | 8.3 and RQ3: "a baseline measurement on audited sets … made that defect visible" | consistent; the abstract says "not by reading weights" |
| Vision-language adapter sentence | stated | A3, A4 | partly (see §9 on "invisible one is not") |
| "The result is largely negative for editing weights as an improvement method" | stated | 9.2: "the positive scenario of the blog post is neither confirmed nor refuted" | tone tension (low); already flagged as O13 in `paper/review/factcheck_2026-11-07.md` |

---

## 12. Open item: combined-v1 line count 5,243 vs 5,441

**Verdict: 5,441 is right; 5,243 is not a count of anything.** Evidence:

- `docs/c4/split.json` gives `train_lines` = 4,927 and `val_lines` = 514, so 4,927 + 514 = **5,441** (`C4_RETRAIN.md` B.1: the trainer's `lines_written` for the job; `tests/test_c4.py` asserts it).
- The same files, counted by project (`docs/c4/train.jsonl` + `val.jsonl`, `doc` field): Hertwig (docId 13922499) = **1,984** lines; Schlussredaktion (docId 13922579) = **3,457** lines. 1,984 + 3,457 = **5,441**.
- The Hertwig figure of 1,786 in Table 1 is the **training side only** of the `hertwig-v2` job (`SEED_NULL.md` L15: the trainer's own split was 1,786 / 198 lines, 1,786 + 198 = 1,984). The 3,457 for Schlussredaktion is the **whole project** (all 118 pages, 107 train + 11 val in the combined job).
- 5,243 = 1,786 + 3,457 adds a training-only count to a whole-project count. The "unexplained difference" of 198 lines is exactly the Hertwig validation side.

Caveats: Hertwig's 1,984 comes from the 22 Oct export, whereas `combined-v1` was trained on the 25 Oct export (wrong images). The total still equals the trainer's `lines_written`, which suggests identical line counts, but the 25 Oct export's own count was not measured here. The Schlussredaktion job's own `lines_written` is not in the repo.

**Suggested fix (for the author):** in Table 1 give Hertwig as "1,984 lines (1,786 train + 198 validation)" and remove the `[TODO: ask the trainer]` note, or keep a one-line trainer query only for the 25 Oct export.

---

## 13. Pre-registered keep rule and registered wording (`docs/PROTOCOL.md`)

| Point | Manuscript | PROTOCOL / sources | Verdict |
|---|---|---|---|
| Keep rule = paired interval excludes 0, `min_effect` 0 by default | 7.7, 9.2 | PROTOCOL §3 (as of 6 Nov) | matches. 9.2's "applied afterwards would have rejected both edits of the first session" ✓ (PROTOCOL §3 says so). |
| Donor rule and "practice sets should be lines the subject was not trained on" | 7.7 applies "the donor rule" as established | PROTOCOL §2: "**PROPOSED AMENDMENT (07.11.2026, not yet adopted)**"; manuscript Appendix B: "proposed" | wording conflict (medium): 7.7 uses it as a fixed rule, while the protocol and Appendix B call it a proposal. Appendix B should say it was applied from 7 Nov on the validation sessions. |
| Appendix B (changes after results were seen) | lists six dated changes | missing: held-out of family 2 cut again to 1,718 lines (`blb-eval-hand-b`, 7 Nov); line-level → page-clustered intervals (7 Nov); R12's minimum-effect rule (8 Nov); adapter keep rule's extra "pages at the cap not up" condition | incomplete (medium) |
| 7.10 rule "written and committed before these scores" | stated | prereg text: v1's Reichenau result already known | overstated, see §6 |
| 7.10 "not changed afterwards" | stated | R12 successor rule | misleading by omission, see §6 |
| Sessions "pre-specified" | abstract, conclusion | Session 1 ran under the earlier keep rule and was re-scored under the later one | acceptable (stated in Appendix B and PROTOCOL §3) |

---

## 14. Ranked mismatches

**High**

1. **Table 10 (7.7), rows 2–3, page intervals.** Printed `[−0.72, −0.06]` and `[−0.67, +0.04]`. Sources: `[−0.84, +0.04]` and `[−0.84, +0.19]`. The first is the withdrawn one-stage figure (CONTROLS_REPORT L5), the second is in no source. As printed, row 2 shows a page-level interval excluding zero, against Table 12, 7.8, RQ2, the abstract, the conclusion and §10.
2. **RQ2 final sentence** "a retraining on corrected data was not run" against 7.11, the abstract, the conclusion and the previous sentence.
3. **Latin v3 / R12 omitted** from the abstract, 7.10, RQ2 and the conclusion. "Neither qualifies" and "not changed afterwards" are right for R9's rule only. A later rule, with minimum effects taken from R9's numbers, qualifies v3. v1's own-collection result was known when R9's rule was written. All three held-out sets were read by three candidates.

**Medium**

4. **7.8 / 7.7 / RQ2 "none of the steps excludes zero"** against the full step (α = 1.0) page interval `[+0.05, +2.19]` in Table 12. **"Clearly worse from 0.75 on"** (7.8): the α = 0.75 intervals contain 0 (source: "clear by α = 1").
5. **Fable 5.1 results** (7.7, RQ2, 9.2): run folders and `CONTROLS_REPORT` text are only on `origin/claude/article-controls-8803eb`. `main` has RESULTS R7 only, and `CONTROLS_REPORT.md` on `main` says no Fable attempt exists. Merge or state the provenance.
6. **"Gradient training is the better tool"** (RQ2), "better at a larger step" (abstract, §10, conclusion). Arm A is "equal" by the preregistered rule. Arm B is secondary, and its page interval against interpolation reaches 0 (seed 2 just excludes it). No uncorrected-pairs control.
7. **Adapter "a localised invisible one is not [repaired]"** (abstract, 9.3, conclusion). p6 was visible to the rule list and covered eight modules, and the keep rule refused its repair. The four single-module defects did no damage. The doubled-α pack for arms (a)/(b) is not disclosed.
8. **Table 11 k11 row** gives "null" in a table captioned "share of the damage recovered". The sources give 100 / 100 / 102 / 74 %.
9. **Donor rule and Appendix B.** 7.7 treats the donor rule as adopted; PROTOCOL and Appendix B say "proposed". Appendix B lacks the 1,718-line held-out redefinition, the page-clustered interval switch and R12's rule.
10. **"A rule written and committed before these scores"** (7.10) — v1's Reichenau scores were known when the rule was written.

**Low**

11. Cross-references swapped: text says "Table 11" for the diffuse table (it is Table 10) and "Table 10" for the validation table (it is Table 11).
12. "0 of 28 edits" (Table 10 row 1): run logs give 27 (7 + 7 + 5 + 8). The report's own text says 28.
13. 7.6 says line-level intervals; sources say class decisions use page-clustered intervals (no class changed).
14. "4,374 held-out lines of 60 pages": the lines are on 58 pages.
15. "Three earlier statements need correction" lists two (§10 says two).
16. 9.2 "all six that kept anything" vs RQ2/7.7 "seven attempts": by the run summaries all seven (a1, a2, a3, o1, h1, f1, replay) kept something. Six excludes the replay.
17. 7.11 "3 to 5 minutes" (source: 2 min 41 s to 4 min 49 s); "seeds differ by 0.01 to 0.02" are standard deviations (range 0.04 in arm B).
18. 7.7 "exactly the half-step row" (close, not exact); 9.3 "late layers" (top layers include 0 and 3); 9.3 undisclosed extra keep-rule condition; "its parent and the donor `hertwig-v2`" are one model.
19. **43 vs 30, a source conflict.** `MODEL_FAMILY.md` says weights for all 43 were obtained (33 downloaded + 4 bases + 6 in-house). `ZOO_SURVEY.md` says only 30 of 37 were obtained and seven were refused (HTTP 429). The manuscript (6.4) states both without reconciling them. The distance figures (0.1–0.3 vs 1.41) and "11 of 13" belong to the 30-model survey but are attributed to "the zoo of 43" in 9.1 item 6 and RQ1.

## 15. Not covered

Sections outside the scope above (1–6, 7.1–7.5, 8.1, 8.3, 9.1–9.2 apart from the cited numbers, 10, appendices apart from B). Reference accuracy. Any recomputation from weights or per-line readings. The Fable run directories were read from the branch, not re-executed. `paper/review/factcheck_2026-11-07.md` is an earlier check by another session; I did not re-verify its items.
