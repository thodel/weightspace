# 2026-11-11 · The planted-defect controls with a Fable programmer

**Type:** control (another programmer model) · **Runs:** `runs/20261111-planted-*-f1-*` (30) ·
**Held-out touched:** once per run · **Source:** `docs/CONTROLS_REPORT.md` §11 ·
Replicates the LLM arms of the [controls](2026-11-07-e-controls.md) · **Result:** [R7](../results.md#r7)

## Plan

The planted-defect positive controls ran on 07.11. with Claude Sonnet 5.5 programmers. Here
they were repeated with Claude Fable 5.1, so that the control is not tied to one model. Only
the LLM arms were rerun: the programmer with passes (a) and the blind programmer with no pass
(b). That covers the kraken corruptions k1–k11, the three variants with an unrelated donor,
and adapter p3, 30 runs in all. The fixed repair list and the bias search do not depend on the
model and were not rerun.

Everything else stayed the same: the corrupted subjects, practice sets, budgets, keep rule,
held-out sets and outcome classes. Each programmer was a fresh Fable subagent that read only
its brief and its reading pack. The pack sat in a folder whose name does not give the defect
away. No programmer was told the defect or saw the Sonnet runs. As on 07.11., each wrote its
program in advance and never ran it. An Opus 5.5 operator ran the programs and scored
held-out, on CPU for kraken and on card 0 for p3.

## Found

- **The same control result.** Fable repaired every localised defect against the parent
  (zeroed, negated and swapped rows, a halved block, a zeroed conv block), in both arms. The
  blind programmer did as well as the one with passes on these. 14 of the 28 Fable final
  models are byte-identical to the Sonnet ones. Over the 22 kraken cells with a defect:
  - **Fable:** 13 repaired, 4 partly, 5 not;
  - **Sonnet:** 15 repaired, 2 partly, 5 not.
- **Adapter p3:** both Fable arms halve every `lora_B` and reach the Sonnet adapter: 0.517,
  against 0.507 for the clean adapter.
- **Four cells differ:**
  - **Diffuse noise on all weights (k9), with passes:** Fable recovered 24 % of the damage,
    Sonnet 83 %. Fable lowered the blank bias and never tried pulling the network toward
    the parent. The one cell where measuring alone made the difference does not carry over
    to a second model.
  - **Blank bias +2 (k4), blind:** Fable judged the shift natural and left it; 0 %, against
    100 % for Sonnet.
  - **Zeroed `e` row, unrelated donor only (k1-u), blind:** Fable raised that row's bias
    instead of copying the donor's row; 43 %, where every other arm got 0–19 %.
  - **Zeroed conv block, unrelated donor only (k8-u), with passes:** Fable recovered 15 %,
    Sonnet 1 %. Neither repaired it.
- **A reading trap in the pack:** the most negative output bias is the no-break space. The
  pack prints it like a plain space, and several Fable programmers took it for a planted
  space defect. The keep rule refused every such edit that was measured.

## Changes

The positive control holds for a second programmer model. The claim that "only the
programmer who measures repairs diffuse damage" holds for one model of two, with one attempt
each. The variance across attempts is still not measured.

## Second p3 attempt (13.11.)

Two more fresh Fable programmers got the p3 pack under a new folder name (`q02`) and were told not to read the first attempt. A watcher ran them on card 0 on 13.11., 20:56–21:17 CET, once the card was free. Both again found the doubled `lora_alpha` and halved every `lora_B`. The programmer with passes then tried a spectral check against the siblings (nothing flagged) and a further ×0.8, which the keep rule refused. Both arms end on the same adapter as attempt 1 and as Sonnet (0.517 on the score set, cached). On p3, at least, the result does not vary across attempts. Runs: `runs/20261113-planted-adapter-p3-alpha2-*-f1-*`.
