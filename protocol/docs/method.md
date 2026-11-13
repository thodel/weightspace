# Method and rules

The rules below are the protocol in force. Changes are listed with their date
and reason under [Protocol amendments](amendments.md); a session is always
judged by the rules that were in force when it was preregistered.

## The loop

1. Load a model's weights into an isolated Python sandbox (no network, no
   training data, no optimiser).
2. **Read**: hypothesise from the tensors what a component does.
3. Spend a **forward pass** on the practice set when a hypothesis needs a test.
   The number of passes per session is the **budget** (0 / 10 / 25 / 50).
4. Write an **edit** in code.
5. **Keep** the edit only if it wins on the practice set (keep rule below).
6. **Freeze**, then evaluate **once** on held-out data the programmer never saw.

## Units

| unit | definition |
|---|---|
| subject | one model artefact, identified by the sha256 of its tensors |
| session | one programmer, one subject, one practice set, one budget, one held-out evaluation at the end |
| pass | one forward pass over the whole practice set; partial passes count as whole |

## Preregistration

Before the first pass, each session records:

1. subject and digests (and those of any donor model);
2. practice set: source, selection rule, size, sha256 of the line list, and
   which side it comes from — training side for sessions up to 7 Nov 2026,
   lines the subject was not trained on since (proposed as
   [A3](amendments.md#a3));
3. held-out set: identity, size, sha256, and an overlap audit against the
   practice set and every corpus the subject was trained on — zero shared
   documents, or the session is labelled *not held-out*;
4. the budget;
5. every hypothesis, with a **predicted direction and magnitude** of CER
   change and the edit that tests it;
6. the controls to be run.

## Keep rule

An edit is kept only if the practice CER falls against the *last kept state*
**and** the paired per-line bootstrap 95 % interval of that difference lies
entirely below −*min_effect* (preregistered per session, default 0 — i.e. the
interval must exclude 0). The collapsed-line count must not rise. The harness
enforces this: a refused keep is undone and logged with its reason.

*(In force since 6 Nov 2026; see [amendments](amendments.md#a1).)*

## Controls

| control | what | question it answers |
|---|---|---|
| C0 no-op | load, save, score | does the round-trip preserve the model; the baseline |
| C1 parent | the parent model on the same sets | what the fine-tune is worth |
| C2 average | uniform average of sibling fine-tunes | the merging baseline |
| C3 α-sweep | parent + α(child − parent), whole model or per block | the cheapest weight-space family |
| C4 retrain | 5 epochs, lr 1e-5, same data | the cheapest gradient alternative |
| C5 sham | random edit of the same tensors with the same Frobenius norm, 5 seeds | does the *content* of the edit matter |
| C6 blind | same programmer and prompt at budget 0 | what the passes bought |

C2–C4 run once per model family; C5 and C6 for every session that reports a
gain.

## Metrics and statistics

- Character and word error rate (CER, WER) on NFD-normalised text with
  whitespace flattened, as kraken's own evaluation counts them; length ratio;
  top-10 confusions; for CTC models the share of frames emitting blank.
- **Collapsed line:** hypothesis under half or over twice the reference
  length, or a repetition loop — the most frequent character trigram at least
  six times and at least twice as often as in the reference, plus two
  ([A2](amendments.md#a2); before 7 Nov 2026 any trigram six times counted).
- For page-level models (adapters): CER, CER with the reading clipped to the
  reference's length, length ratio, pages that run to the token cap, loops.
- Every difference: paired per-line bootstrap 95 % interval, 2,000 resamples,
  seed 20261106; since 7 Nov 2026 also clustered by page, and a claim needs
  the page-level interval ([A4](amendments.md#a4)).
- **Held-out sets are chosen by data properties only**, never by how a model
  reads them.
- All sessions are reported, including reverted and failed ones. The ratio of
  kept to attempted edits is itself a result.

## Artefacts per session

Preregistration, the harness log (every hypothesis, edit, pass and decision),
the programmer's notes, a sparse diff against the starting weights, a summary
with digests, set hashes, budget used, metrics, controls and deviations, the
exact package versions and the sandbox command line, and the programmer's
model and settings. Weights are not published; diffs are.
