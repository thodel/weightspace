# Plan: the programmer replicates with Gemini 4 Argon and a GPT model (drafted 15.11.2026)

Status: **prepared, not run.** Everything is in place except the two API keys and the GPT model id.
Owner: Tobias Hodel. Operator: a Claude session. Related: `docs/PHASE5_FABLE_REPORT.md`,
`docs/KRAKEN_S2_FABLE.md`, `protocol/docs/journal/2026-11-11-c-planted-fable.md`,
`docs/epics/EPIC_LLM_PLAYBOOK.md` (deliverable 5: "any further provider only if credentials exist").

**The programmer must not read this file.** §6 records what the operator knows. `wsp.programmer`
keeps it out of reach: the tool layer opens only the files the brief lists.

## 1. Question

The Fable replicates swapped the programmer and kept everything else fixed. On the phase-5 adapter,
Fable 5.1 kept exactly the edit Opus 5.5 had kept, and the frozen adapter was byte-identical. On kraken
session 2, both programmers moved toward the parent, but Fable's smaller step held up on held-out. On the
planted defects, the control result held, with four cells that differed.

All four programmers so far are Claude models. **Do models from other vendors, given the same brief, the
same reading and the same rules, choose the same edits and reach the same held-out results?** If they do,
the edit choice is set by the reading and the rules, not by the model family. If they do not, the
differences tell us which parts of the result belong to the programmer.

## 2. What is run

"The same experiment" means the Fable replicates, with the programmer swapped again: once for
**Gemini 4 Argon** and once for **a GPT model**. Each vendor gets one attempt per session, as Fable did.

| id | session | template (Fable run) | engine | cost per vendor | priority |
|---|---|---|---|---|---|
| **E1** | phase-5 adapter, `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` | `runs/20261110-kurrent-page-v2-f1-b10-1` | card 0, `evaluate_qlora` | ≤ 10 passes × 35 min, plus held-out 7.8 h (see §3.3) | **primary** |
| **E2** | kraken session 2, `kraken-bern-combined-v1` | `runs/20261111-bern-combined-v1-f1-b10-1` | CPU | ≤ 10 passes × 1 min, held-out minutes | run first, as the shakedown |
| E3 | planted-defect controls, arms a and b, k1–k11 plus the `-u` variants and adapter p3 | `runs/20261111-planted-*-f1-*` (30) | CPU; card 0 for p3 | 30 runs; blind arm b has no pass | after E1 and E2, only on a go-ahead |

The run ids follow the Fable pattern: `f1` becomes `g1` for Gemini and `o1` for OpenAI. For example,
`runs/<date>-kurrent-page-v2-g1-b10-1` and `runs/<date>-bern-combined-v1-o1-b10-1`.

**Order:**
1. **Gate 0:** `wsp.programmer check` for both providers (§4.1).
2. **E2:** both vendors. This is CPU only, about an hour each. The two vendors can run in parallel because
   each has its own run folder. E2 also tests the tool layer on a real brief.
3. **E1:** Gemini, then GPT, one after the other on card 0, each pass only when `wsp.card.ready` allows it.
4. **E3:** only on the user's go-ahead. Its runner (`scripts/run_planted_f1.py`) writes each program in
   advance; it needs a `--programmer` switch that calls `wsp.programmer` for one round instead of a
   Fable subagent. That switch is not written yet.

## 3. Set-up: what stays the same and what differs

### 3.1 The same as the Fable runs

- The subject, its file sha256 and tensor digest. `xv_prepare` refuses to continue if the new
  `session.json` differs from the template.
- The practice list and its sha256.
- The held-out set, scored once after freeze.
- The budget: 10 passes, the first being the baseline C0.
- The keep rule:
  - E1: the paired page bootstrap, plus the condition that pages at the cap must not rise;
  - E2: the paired line bootstrap, plus the collapsed-line condition.
- The seed: 20261106.
- The engine, the sandbox (`env -i`, `unshare -Un`, CPU or card 0) and the operator's procedure.
- The brief: the template's `PROGRAMMER_BRIEF.md`, word for word, with only the run name and the
  checkout path replaced. E1 also gets the errata of §3.4.
- The pack (E2) and the readings the brief lists.
- §1–4 and §6 of the pre-registration, with only the run name and path replaced. §5 is left empty for
  the programmer.
- Rounds: round 0 writes §5, `program_01.py` and `notes.md`. In each later round the programmer gets the
  previous program's stdout, then writes the next program. The last program ends with `s.freeze()`.

### 3.2 What differs, and why

| | Fable 5.1 | Gemini / GPT |
|---|---|---|
| How it runs | Claude Code subagent | `python -m wsp.programmer round …`, called through the vendor's HTTP API |
| Tools | Read, Write, Edit, Bash | `read_file`, `list_dir`, `write_file`, `run_python`, `finish_round`. Their descriptions map one-to-one onto what the brief says ("read", "write", "CPU reading script", `py_compile`). |
| Reading limits | by instruction only | by instruction **and** enforced: `read_file` opens only the files in `programmer.json`, which are the files the brief lists, and the run folder. The `programmer/` transcript folder is excluded. Refused attempts are logged and reported. |
| Writing limits | by instruction | enforced: only §5 of `preregistration.md`, `program_NN.py` (checked with `py_compile` on write) and `notes.md` |
| CPU reading | Bash in the checkout | `run_python` in the same sandbox as the programs, on a **throw-away copy** of the run folder without `passes/` and the work adapter. Code that calls `measure`/`keep`/`freeze`, runs `program_NN`, scores held-out or starts processes is refused. |
| Memory between rounds | the same subagent, resumed | the same conversation, re-sent from `programmer/conversation.json` (Gemini's thought signatures included) |
| Reasoning effort | default | the provider's default. `programmer.json` `options` stays `{}` unless the user decides otherwise. |
| Record | the subagent's notes | the programmer's `notes.md`, plus `programmer/transcript.jsonl` (every request, tool call and result), `usage.json` (input, output and reasoning tokens) and `rounds.json` |
| Pre-registration §7 | operator's text about Fable | rewritten by `xv_prepare`. It names the vendor, the pinned model id and the tool layer, and points to §6 of this file for the operator's knowledge. It names no earlier edit or result (the lesson of attempt 0 in `docs/KRAKEN_S2_FABLE.md` §1). |

- **Images (E2):** the Bern brief allows the practice line images. Fable only read their sizes with
  Python. `run_python` can do the same, but no image is sent to the model. That matches what Fable
  actually did.
- **One system message:** the tools need a system message that maps the brief's verbs to the tools.
  It is `SYSTEM` in `wsp/programmer.py`. It contains no domain content.

### 3.3 Held-out on card 0 (E1)

The engine is deterministic. 32 of 32 pages repeated byte for byte, and Fable's 733 held-out readings
matched the Opus session's in all 23 chunks. So:
- if a frozen adapter's file sha256 equals one already scored (`3f68caa5…`, or the unedited v2 when
  nothing is kept), its held-out readings are reused, and the report says so;
- every other frozen adapter is scored once with `wsp.run heldout --set kurrent-test`, which takes
  about 7.8 h on card 0, only when `wsp.card.ready` allows.

### 3.4 Errata in the E1 brief (a deviation, decided here)

Fable found two errors in the E1 brief before its first pass. Neither affected a pass:
- the API example's `axis=0` for `project_out` on `lora_B` raises; `axis=1` is right;
- `practice_per_page` holds `[edits, reference characters]`.

For Gemini and GPT the brief carries the corrections. Leaving them in would test error-finding, which is
not the question here. Each pre-registration records this as a deviation from the template. The E2 brief
had no errata.

## 4. Procedure for each run

### 4.1 Keys (the user)

The keys never go into the repository, a run folder, a transcript or a log. `wsp.programmer` reads them
from the environment or from a file that only you can read:

```bash
mkdir -p ~/atr-cache/wsp/keys && chmod 700 ~/atr-cache/wsp/keys
```

```bash
printf '%s' 'PASTE-GEMINI-KEY' > ~/atr-cache/wsp/keys/gemini.key && chmod 600 ~/atr-cache/wsp/keys/gemini.key
```

```bash
printf '%s' 'PASTE-OPENAI-KEY' > ~/atr-cache/wsp/keys/openai.key && chmod 600 ~/atr-cache/wsp/keys/openai.key
```

Alternatively, set `GEMINI_API_KEY` and `OPENAI_API_KEY` in the environment of the operator's shell.
A key file readable by others is refused.

**Gate 0 (operator):**

```bash
.venv/bin/python -m wsp.programmer check --provider gemini --model gemini-4-argon
```

```bash
.venv/bin/python -m wsp.programmer check --provider openai --model <gpt id>
```

`check` confirms four things and prints no key:
- the key is found;
- the model is in the provider's model list (if not, it prints the candidates);
- one tool call comes back;
- which model version served it.

The served version is the id that goes into `programmer.json` and the report.

### 4.2 One session (operator)

1. Start from a checkout of `main` that contains the template runs, on a new branch.
2. Set up the run folder:

   ```bash
   .venv/bin/python scripts/xv_prepare.py --experiment bern --provider gemini --model gemini-4-argon --label "Gemini 4 Argon"
   ```

   It creates `runs/<date>-bern-combined-v1-g1-b10-1/` with `session.json`, the brief, the pack, the
   pre-registration (§5 empty) and `programmer.json`. Use `--dry-run` first.
3. Read the brief and the pre-registration once before round 0. Check that neither names an earlier
   session's edit or result. Commit them: "before the first pass".
4. Round 0:

   ```bash
   .venv/bin/python -m wsp.programmer round --run runs/<id> --provider gemini --round 0
   ```

   Check that §5, `program_01.py` and `notes.md` exist. Commit them before the first pass.
5. Run `program_01.py` in the sandbox, exactly as for Fable (`wsp.run.sandbox_command`, stdout to
   `stdout_program_01.log`). For E1 that means card 0, only after `wsp.card.ready`.
6. Round n:

   ```bash
   .venv/bin/python -m wsp.programmer round --run runs/<id> --provider gemini --round 1 --stdout runs/<id>/stdout_program_01.log
   ```

   Repeat until a program has frozen the session or the budget is used up.
7. Score held-out once (§3.3), run the controls (§5), and write the report.

If the model answers without a tool call, the loop itself sends one fixed reminder ("Continue the round
with your tools …"), and the transcript records it. If a round still ends without `finish_round` (the
exit code is 2), the operator may run the round **once** more. That is logged as a deviation. The
operator never writes or repairs a program.

### 4.3 What is sent to the vendors

Each round sends the following to Google or OpenAI:
- the brief;
- the files the programmer chooses to read: the readings or pack, the C0 baseline, the **practice list
  with its reference transcriptions**, and the harness source;
- the output of its own reading scripts and of the programs.

No held-out data, no model weights, no keys and no images are sent. The transcriptions of the E1
practice pages come from the val side of `german-kurrent-xvi-xix-v1`; those of E2 come from the Bern
Disputation training pages.

**Providing the keys counts as the user's go-ahead to send this material to the two vendors.** If any of
it may not leave the machine, say so before Gate 0.

## 5. Controls and the report

For each kept edit, the same controls as in the Fable run of that session:
- **E1:**
  - C5 shams (seeds 20261108, 20261110 and 20261111 already exist);
  - C0, C1, C3 and the decode comparator are reused, because they do not depend on the edit.
- **E2:**
  - C5 sham, 5 seeds, the same Frobenius norm;
  - the split of the kept diff, if it has more than one part;
  - C0 and C1 from `docs/baselines/`.

Reports:
- `docs/XVENDOR_REPORT.md`, with one section per session. Each holds:
  - the pass table;
  - held-out with page and document intervals (A4);
  - the controls;
  - a **comparison of four programmers**: Opus 5.5, Fable 5.1, Gemini 4 Argon and GPT.
- For each programmer the report also records:
  - the pinned and the served model id, the date and the settings;
  - passes used, edits tried and edits kept;
  - tokens (input, output, reasoning) and tool calls;
  - the number of `run_python` scripts;
  - refused reads and writes, with their paths;
  - nudges.
- Journal entries in `protocol/docs/journal/`, and a RESULTS paragraph.

What would count as a finding, stated before the runs:
- **E1:** a vendor that keeps direction 0 of `layers.31.self_attn.o_proj` reaches the same adapter (sha
  `3f68caa5…`) and the same held-out null. Two more programmers on the same direction make "the reading
  decides the edit" stronger. A different kept edit needs its own held-out score and shams. It is judged
  against the sham spread already measured (−1.3 to −5.7 points on practice).
- **E2:** whether the edit again moves toward the parent, and whether it beats the C0 held-out with a
  page interval that excludes 0, as Fable's did and Opus's did not.
- **Not a finding:** a practice gain alone. The shams show that the keep rule can keep a random direction
  in E1.

## 6. What the operator knows (programmer must not read)

Before the first pass the operator knows every earlier result of the sessions it replicates:
- **E1:** both the Opus session and the Fable session kept direction 0 of
  `layers.31.self_attn.o_proj`, removed from `B`. Practice CER went 0.4515 → 0.3470 and pages at the cap
  9 → 4. Held-out was +0.24 points [−3.5, +3.8], a null. Fable's frozen adapter is byte-identical.
- **E2:**
  - Opus transplanted the parent's output layer: held-out +0.15, not a gain.
  - Fable lowered the blank bias twice, then moved 12.5 % toward `hertwig-v2`: held-out −0.57
    [−1.15, −0.05]. The gain came from the interpolation; the blank steps did not transfer.
- **E3:** the outcome class of every Sonnet and Fable cell (`docs/controls/planted_f1.md`).

None of this goes into a brief, a pre-registration or a round prompt. The round prompts are fixed in
`wsp/programmer.py` (`round_prompt`) and carry only the program's stdout. The operator sends no other
message to the programmer, apart from the one allowed nudge.

## 7. Decisions for the user

1. **The GPT model id.** "A GPT model" is not yet pinned. `check --provider openai` lists the ids the key
   can use; pick one, ideally the vendor's current flagship. The id is recorded as served.
2. **Gemini id:** `gemini-4-argon`. Public sources give this as the stable id since 13.10.2026; Gate 0
   confirms it against the key's model list.
3. **Sending data** (§4.3): providing the keys counts as the go-ahead.
4. **E3:** in or out. It costs 30 runs per vendor and needs the runner switch of §2.
5. **One attempt per vendor**, as for Fable. A second attempt, `g2`/`o2`, only if the first is
   inconclusive because of a nudge or a tool failure.

## 8. What was prepared (15.11.2026)

- `wsp/programmer.py`:
  - the provider clients (Gemini `generateContent`, OpenAI Chat Completions) over plain HTTP, with no
    new dependency;
  - the tool layer with enforced read and write limits;
  - the round loop with a persisted conversation;
  - token accounting;
  - `check`.
- `scripts/xv_prepare.py`: builds a run folder from a Fable template (E1, E2), with the checks of §3.1.
- `tests/test_programmer.py`, all without network:
  - read and write rules, and the §5-only rule;
  - the throw-away sandbox;
  - rounds carrying over and the round order;
  - key handling;
  - both clients against a local fake HTTP server, including Gemini's thought signatures.
- Not yet done:
  - the E3 runner switch;
  - a real call to either API, because there are no keys yet.
