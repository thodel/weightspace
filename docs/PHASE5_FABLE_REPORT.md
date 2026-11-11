# Phase 5 with a second programmer: Fable 5.1 finds the same direction, and held-out still says nothing (asteraix, 10.–11.11.2026)

Session `runs/20261110-kurrent-page-v2-f1-b10-1/`. It replicates
`runs/20261107-kurrent-page-v2-b10-1`, the Opus 5.5 session of `docs/PHASE5_REPORT.md`, with
**Claude Fable 5.1 (`claude-fable-5-1`)** as programmer. Everything else is the same: subject,
practice set, held-out set, budget, engine and keep rule. Everything ran on card 0 through the
trainer's own `evaluate_qlora`. Nothing was written to the share, the registry, HF or the
trainer.

**In short:** working without sight of the Opus session, Fable kept exactly one edit, the
same one Opus kept: direction 0 of `layers.31.self_attn.o_proj`, removed from `B`. Its frozen
adapter is **byte-identical** to the Opus session's (sha256 `3f68caa5…`). Its practice
readings of passes 1–3 and all 733 held-out readings repeat the Opus session's byte for byte:
held-out **+0.24 points [−3.5, +3.8] by pages, [−5.0, +6.7] by documents**. The two
programmers agree on the direction, and the result stays a held-out null.

Two new random-direction shams sharpen the phase-5 conclusion. One of them would itself have
passed the keep rule: −5.7 points [−11.8, −1.2], at the cap 9 → 7. H1 is not separable from
two of the three shams.

## 1. Set-up

- **Subject:**
  - `qwen3.5-4b-german-kurrent-xvi-xix-page-v2`, tensor digest `fb4f55ee…`, file sha256 `1b70ce4a…`;
  - peft LoRA, r 64, α 128, 128 language-tower modules;
  - only the adapter's work copy in the run folder was edited.
- **Practice:** `data/practice/kurrent-val-64.jsonl`, 64 val pages, sha256 `2cc5278e…`.
  C0 reads CER 0.4515, `truncated_cer` 0.2564, 9 pages at the cap.
- **Held-out:** `kurrent-test`, 733 pages from 70 documents (`c56698a7…`), scored once after
  freeze with `wsp.run heldout`.
- **Budget and rule:**
  - budget 10, baseline first, `min_effect` 0;
  - PROTOCOL §3 on the paired page bootstrap, plus the condition that pages at the cap must not rise;
  - claims carry the document-clustered interval (amendment A4). Here the unit is the page,
    so the cluster is the document.
- **Programmer:**
  - Claude Fable 5.1 as a Claude Code subagent (`model: fable`), default effort, tools
    Read/Write/Bash for CPU-only reading;
  - it read only `PROGRAMMER_BRIEF.md` and what that file lists: the budget-0 reading
    `docs/readings/adapter_kurrent-page-v2.*`, C0, the practice list and the harness source;
  - it did **not** read the phase-5 plan, the Opus pre-registration, or any of the Opus
    session's files;
  - it wrote `program_01.py` to `program_03.py` in rounds. Each round started after the
    stdout of the previous program.
- **Operator:**
  - a Claude Opus 5.5 session ran the programs and did the held-out score, the controls and
    this report;
  - it had read the phase-5 plan, the Opus pre-registration and §1–3 of the phase-5 report
    beforehand, so it knew the kept edit (said so in the pre-registration, §7);
  - the brief names no edit and no result of the earlier session.
- **Deviation from the task as written:** the task asked the session itself to run "as
  Claude Fable 5.1". The session was running on Opus 5.5, so the programmer role went to a
  Fable 5.1 subagent, as in the f1 attempts of 09.11. (`runs/20261109-val-*-f1-b10-1`). The
  pre-registration names both models and their roles.
- **Two errors in the brief, found by the programmer before the first pass:**
  - the API example's `axis=0` for `project_out` on `lora_B` is wrong, and `axis=1` is right;
  - `practice_per_page` holds `[edits, reference characters]`.

  Neither affected a pass. Both are recorded in `notes.md`.
- **Harness changes (this branch):**
  - `reference_runner` continues its pass numbering across programs. Before, a second
    program overwrote `pass-001`.
  - `wsp.vlm_reference.HOME` and `wsp.card.STATE` now find the account's home as the first
    ancestor that holds `atr-cache`. Before, a worktree under `.claude/worktrees/` sent both
    to a wrong path, and the sandbox saw no watcher state.

## 2. The session (budget 10, 5 used, 10.11.2026 18:30–23:33 CET)

| pass | edit (Fable's label) | practice CER | `truncated_cer` | at cap | against last kept | decision |
|---|---|---:|---:|---:|---|---|
| 1 | none (C0) | 0.4515 | 0.2564 | 9 | — | — |
| 2 | **H1:** direction 0 of `layers.31.self_attn.o_proj` out of `B` | **0.3470** | 0.2341 | **4** | −10.45 [−20.3, −2.4] | **kept** |
| 3 | H2: direction 0 of `layers.31` and `layers.27.mlp.down_proj` | 0.3922 | 0.2346 | 6 | +4.52 [−2.4, +15.8] | refused |
| 4 | H5: directions 1–2 of the same `o_proj` as well | 0.3378 | 0.2326 | 4 | −0.92 [−7.6, +4.7] | refused |
| 5 | H3: q/k/v/o `B` of layers 23, 27 and 31 × 0.5 | 0.3859 | 0.2608 | 6 | +3.89 [−1.4, +11.1] | refused |

- **Hypotheses:** every one was pre-registered with its predictions before its pass
  (`preregistration.md` §5, `log.jsonl`).
- **H1:** the prediction was the cap 9 → 7 or fewer and CER −3 points (range −8 to +1). The
  cap was met; CER fell more than predicted, because five cap pages stopped.
- **H5:** one cap page stopped (`011629`) and another returned to the cap (`008159`). The
  programmer read it as a chaotic swap, not an effect.
- **Freeze:** Fable froze at 5 passes, with 5 left unused on purpose. Its pre-registered
  rule 4 says not to spend passes on variants of a refused family. It ruled out a dose
  beyond 1, because that would add a direction the adapter never learned.

## 3. Held-out (once, 11.11.2026, finished 09:26 CET, 7.8 h on card 0)

| 733 test pages | CER (trainer) | CER (wsp) | `truncated_cer` | `length_ratio` | at cap | loops |
|---|---:|---:|---:|---:|---:|---:|
| v2 unedited (phase 5, same card and engine) | 0.4053 | 0.4012 | 0.2430 | 1.177 | 44 | 63 |
| Fable session, frozen | 0.4079 | 0.4036 | 0.2460 | 1.175 | 43 | 61 |

- **Fable − v2:**
  - **+0.24 points**, page bootstrap **[−3.5, +3.8]**, document-clustered **[−5.0, +6.7]**;
  - 225 pages better, 233 worse (`docs/baselines/p5f-compare.json`).
- **Against the Opus session:** all 23 chunks of raw readings are byte-identical, as the
  identical adapter and the deterministic engine predict. So the Opus session's held-out
  score reproduces exactly, and this is not a second, independent held-out data point.

## 4. Controls on the practice pages

| practice, 64 pages | CER | `truncated_cer` | at cap | − C0 (page [doc] interval) | H1 − it |
|---|---:|---:|---:|---|---|
| C0, v2 | 0.4515 | 0.2564 | 9 | — | −10.45 [−20.3, −2.4] [−20.4, −2.8] |
| **H1 (both sessions)** | **0.3470** | 0.2341 | **4** | −10.45 | — |
| C5 sham seed 20261108 (phase 5) | 0.4128 | 0.2525 | 7 | −3.87 [−9.1, +0.1] | −6.57 [−16.3, +0.9] |
| C5 sham seed 20261110 (new) | 0.3947 | 0.2517 | 7 | **−5.68 [−11.8, −1.2]** | −4.77 [−13.4, +1.6] |
| C5 sham seed 20261111 (new) | 0.4382 | 0.2545 | 9 | −1.33 [−5.6, +2.6] | −9.12 [−20.3, −1.0] |
| C3: whole adapter × 0.75 / 0.9 / 1.1 (phase 5) | 0.398 / 0.423 / 0.504 | 0.261 / 0.261 / 0.271 | 5 / 7 / 6 | | |
| `repetition_penalty` 1.05, no edit (phase 5) | 0.3680 | 0.2287 | 4 | | |
| C1, base without adapter (phase 5) | 1.6237 | 0.5581 | 23 | | |

How the controls were run:
- The shams remove a seeded random unit direction from the same `B`, the way H1 does.
- The document-clustered intervals of the practice rows differ from the page intervals by at
  most 0.4 points: practice has one page per document, apart from two documents with three.
- C1, C3 and the decode comparator do not depend on the edit. They are a function of
  weights, pages and engine, so they are reused from phase 5 and were not run again
  (pre-registration §6).
- C4 was not submitted.

What the shams show:
- **The keep rule would have kept a random direction.** Seed 20261110 lowers practice CER by
  5.7 points with an interval below 0, and the pages at the cap do not rise (9 → 7).
- H1 is better than the mean sham (−3.6 points), but its interval against the sham excludes 0
  for only one of the three seeds.
- On 64 pages, "any perturbation of this module reshuffles which pages loop" explains a large
  part of the practice gain. The phase-5 report inferred this from one sham; three seeds
  show it directly.

## 5. Comparison of the two programmers

| | Opus 5.5 (07.11.) | Fable 5.1 (10.11.) |
|---|---|---|
| passes used | 6 of 10 | 5 of 10 |
| edits tried | 5 | 4 |
| kept | 1: direction 0 of `layers.31.self_attn.o_proj` | **the same**, byte-identical adapter |
| other edits | dir 0 of `layers.31`+`27.mlp.down_proj`; layers 27–31 × 0.75; H1 at dose 1.5 and 0.5 | the same `down_proj` pair; dirs 1–2 of the same `o_proj`; late attention × 0.5 |
| practice CER | 0.4515 → 0.3470, −10.45 [−20.3, −2.4] | identical |
| held-out, page [doc] | +0.24 [−3.5, +3.9] (page) | +0.24 [−3.5, +3.8] [−5.0, +6.7], identical readings |
| prediction for H1 | cap 9 → 5–7, CER −2 to −6 points | cap 9 → ≤ 7, CER −3 points (−8 to +1), `truncated_cer` ±0.5 |

- **Same direction:** both programmers drew H1 from the same budget-0 reading. It is the
  top row of the logit-lens table: z −3.60 for `<|im_end|>`, −5.52 for `\n` and −8.06 for
  `¬`. Their second hypothesis was the same too: the next two rows, the `down_proj`
  directions of layers 31 and 27. Both pairs were tested in the same pass order, and the
  practice readings of passes 1–3 are byte-identical.
- **Different follow-ups:**
  - Opus tested the dose of H1 (0.5, 1.5), which gives a curve with its minimum at 1;
  - Fable tested the specificity of H1 (the module's next two directions) and coarse
    scaling of the late attention.

  Every follow-up of both programmers was refused.
- **What the agreement means:** it is not independent evidence for the edit. The edit is the
  top line of a table that both programmers were given. The agreement shows that the
  reading, more than the programmer, drives the first hypothesis, and that two strong
  models read it the same way. On the step that failed, practice to held-out, the
  programmer makes no difference, because the artefact is the same.

## 6. What this adds

1. **The phase-5 result does not depend on the programmer model.** A second model, working
   blind to the first, reaches the same kept edit and the same held-out null.
2. **The held-out engine is deterministic at full scale.** The same adapter gave 733 of 733
   byte-identical readings two days apart, 7.8 h against 8.2 h of wall time.
3. **The keep rule on 64 pages can be passed by a random direction.** It accepted one of
   three seeded shams. The phase-5 report proposed a noise floor of several seeded sham
   directions per pass; that case is now stronger. Against the best of the three shams, H1's
   interval contains 0.
4. **Not shown:** whether a programmer given a different reading would have found a
   different edit. Both had the same lens table, and its top row decided.

Files:
- `runs/20261110-kurrent-page-v2-f1-b10-1/`: pre-registration, brief, notes, programs, log,
  summary with the held-out score, diff and stdout;
- `docs/baselines/p5f-sham-*.json` and `p5f-compare.json`;
- `scripts/p5f_queue.py` and `p5f_compare.py`.
