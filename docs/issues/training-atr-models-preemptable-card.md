# A preemptable card: let other work use a GPU and yield to queued jobs

Filed 06.11.2026 as thodel/training-atr-models#396. This file is the draft as filed.

## Where we are

Since #12 (22.10.2026) the service runs with `ATR_TRAIN_GPUS=0,1` and allocates a card per job. Its only gate is free VRAM on a card no job of ours holds (`min_free_vram_mb` 12000, `vlm_min_free_vram_mb` 24000). Foreign processes are invisible to it except as missing memory; `GET /gpu` reports them as unaccounted.

That makes the arbitration run the wrong way round for anyone else on the box. A weightspace pass (thodel/weightspace, phase 0) that holds 10 GB on a card queues the next training job behind it until the pass ends. Nothing tells the other process that a job is waiting. The A40s have no MIG and both cards are in Default compute mode, so there is no hardware priority either: whoever allocates first holds the card until they exit.

Today (06.11.2026, 14:30) the picture is inverted on top of that: card 0 is full with `scripts/measure_noise_floor.py`, a manual script pinned to `--device cuda:0`, while card 1 idles. weightspace's own docs still say "card 0 is free, card 1 is the trainer's", which has been stale since #12.

## What a preemptable card would mean

The service keeps both cards. Other work (weightspace sessions, ad-hoc scripts) may use any card with headroom, but **yields when a job of ours is queued and no card has enough free memory**. Jobs of this service are never the ones that wait.

Preemption at pass granularity is enough: a weightspace pass is minutes, kraken training is hours. Nobody needs a kill signal mid-kernel.

## Proposal

Two small things on this side, the rest in weightspace.

1. **Advertise the need.** `GET /gpu` (or `/health`, which is keyless) says whether a queued job of this host is waiting for a card, and how much VRAM it needs: `{"waiting": {"job_id": …, "engine": "kraken", "needs_mib": 12000}}` or `null`. The scheduler already knows both numbers when it decides not to spawn; it only has to write them down. No key needed for a yes/no that leaks nothing.

2. **Name the convention in INFRASTRUCTURE.md, "The cards".** Foreign work may use either card. It checks `waiting` before each unit of work and releases the card when a job is waiting and the card it holds is the one that would fit. The service never kills a foreign process; the VRAM check stays the only thing that refuses a start. A foreign process that ignores the convention is what it is today: an unaccounted process the scheduler waits for.

Optional, later: a soft `ATR_TRAIN_PREEMPT_GRACE_S`. If a job has waited longer than that on a card held only by unaccounted processes, `journalctl` logs who holds it, so the next "why is my job queued" has an answer.

On the weightspace side (not this repo): `wsp.run` picks the card with the most free memory per pass, sets `CUDA_VISIBLE_DEVICES` for the child, polls `waiting` between passes, falls back to CPU when it must yield, and its docs drop "never card 1".

## What this does not do

- No hard reservation. `ATR_TRAIN_GPUS=0` would hand card 1 to weightspace for good; that is a split, not preemption, and it costs the parallel runs #12 bought.
- No signal to foreign processes, no cgroups, no MPS. Cooperative only.
- Manual scripts of this repo (`measure_noise_floor.py` and friends) that default to `cuda:0` keep working as before; a follow-up could give them the same card picker.

## Acceptance

- `GET /gpu` returns `waiting` as above; `null` when nothing is queued or a card is free.
- A test: with both cards "held" (mocked free memory below the threshold) and one queued kraken job, `waiting.needs_mib == 12000`; with a VLM job, 24000.
- INFRASTRUCTURE.md "The cards" updated, and its "Today" paragraph no longer says `ATR_TRAIN_GPU=1` one job at a time.
