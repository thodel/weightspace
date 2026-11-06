# Deploying on asteraix — the command sequence

asteraix = `dhserver03`, 130.92.59.242, SSH alias `asteraix` on the laptop
(`training-atr-models/docs/INFRASTRUCTURE.md`). Checkouts live under `~/Repo/`.
Everything below runs as the normal user there, writes only under
`~/Repo/weightspace` and `~/atr-cache`, and uses card 0 at most.

## 1. First time: bootstrap (about 10 minutes, mostly pip)

From the laptop:

```bash
ssh asteraix
```

On asteraix, once: a key for this repository (the account there has none
GitHub knows; `git@github.com: Permission denied (publickey)` on 06.11.2026).

```bash
ssh-keygen -t ed25519 -N '' -C "asteraix weightspace deploy key" -f ~/.ssh/id_ed25519_weightspace && printf 'Host github.com\n  IdentityFile ~/.ssh/id_ed25519_weightspace\n  IdentitiesOnly yes\n' >> ~/.ssh/config && chmod 600 ~/.ssh/config && cat ~/.ssh/id_ed25519_weightspace.pub
```

Paste the printed public key at
https://github.com/thodel/weightspace/settings/keys → *Add deploy key*, tick
**Allow write access** (the phase0 branch is pushed from there). Then:

```bash
ssh -T git@github.com
```

must answer `Hi thodel/weightspace! You've successfully authenticated…`. Then:

```bash
git clone git@github.com:thodel/weightspace.git ~/Repo/weightspace
```

```bash
bash ~/Repo/weightspace/deploy/asteraix_bootstrap.sh
```

The script runs, in order: `preflight` (share mounted, python3.12, GPUs,
GitHub key), `checkout` (pull), `venv` (kraken 7.0.2 pinned, falls back to
the trainer's torch version if pip's torch does not resolve, writes
`docs/ENV_ASTERAIX.txt`), `tests` (the 18 engine-free tests), `weights`
(copies the six in-house `.mlmodel` files, writes `models/DIGESTS.md` and
checks three of them against the gateway's digests), `parent` (fetches
`cursive_inzigkofen` by DOI through htrmopo), `summary`.

It is idempotent: after a failure, fix the cause and rerun the whole script or
one step, e.g. `bash deploy/asteraix_bootstrap.sh venv tests`.

Expected end of the output:

```
kraken 7.0.2 | torch 2.8.0+cu128 | cuda True | numpy …
18 passed
kraken-bern-combined-v1  35e7eae2…  16396542  match
…
```

A `MISMATCH` against the gateway digest is a finding to write down, not to fix.
A `MISSING on share` line means the trained model moved; `gateway_models`
knows the current `local_path`.

## 2. Commit what the bootstrap produced

```bash
cd ~/Repo/weightspace && git checkout -b phase0 && git add models/DIGESTS.md docs/ENV_ASTERAIX.txt && git commit -m "asteraix: digests and environment" && git push -u origin phase0
```

## 3. Start the phase 0 session

Either a Claude Code session on asteraix in the checkout, opened with:

```bash
cd ~/Repo/weightspace && claude "Read docs/BRIEF_PHASE0.md and execute phase 0 from section 3 on, on branch phase0. Rules: card 0 only, no writes to the share, no held-out data, no budgeted session."
```

or the laptop session working over `ssh asteraix` with the same brief.
`claude` must exist on asteraix (`which claude`); if not, the laptop session
drives it over SSH.

## 4. Per-shell environment for every later session

```bash
cd ~/Repo/weightspace && export TMPDIR=~/atr-cache/tmp CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && source .venv/bin/activate
```

`HF_HUB_OFFLINE=1` goes on **after** the bootstrap (the `parent` step and the
practice-set build may need the hub once).

## 5. The round-trip test, once a practice set exists

```bash
cd ~/Repo/weightspace && WSP_MODEL=models/kraken/kraken-german-xix-blb-v1.mlmodel WSP_PRACTICE=data/practice/blb-train-300.jsonl .venv/bin/python -m pytest -q tests/asteraix
```

## 6. Undo

```bash
rm -rf ~/Repo/weightspace/.venv ~/Repo/weightspace/models/kraken ~/atr-cache/wsp
```

Nothing else was written. The clone itself can stay or go.

## What this does not do

No registry entry, no change to `atr-train` or its venvs, no upload, no
`systemd` unit. Serving a candidate is phase 4 and needs a separate OK.
