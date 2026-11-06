#!/usr/bin/env bash
# Bootstrap this repo on asteraix: checkout, venv, tests, weights with digests.
# Idempotent: rerun after a failure. Writes only under ~/Repo/weightspace and
# ~/atr-cache. Never touches the trainer's venvs, the share, or card 1.
#
#   bash ~/Repo/weightspace/deploy/asteraix_bootstrap.sh            # everything
#   bash ~/Repo/weightspace/deploy/asteraix_bootstrap.sh preflight  # one step
# steps: preflight checkout venv tests weights parent summary
set -euo pipefail

REPO=~/Repo/weightspace
REMOTE=git@github.com:thodel/weightspace.git
SHARE=/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained
PARENT_DOI=10.5281/zenodo.18207767      # kraken-cursive_inzigkofen, parent of the bern family
export TMPDIR=~/atr-cache/tmp
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

MODELS=(kraken-bern-hertwig-v1 kraken-bern-hertwig-v2 kraken-bern-schlussredaktion-v1
        kraken-bern-combined-v1 kraken-german-xix-v1 kraken-german-xix-blb-v1)
# file sha256 the gateway reported on 06.11.2026 (gateway_models → weights_digest.files)
declare -A GATEWAY_SHA=(
  [kraken-bern-combined-v1]=35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6
  [kraken-bern-schlussredaktion-v1]=d26eb02dfaf89f8f222165a1e5c982e38f19a915a5a744e4ad6c15f69b71f526
  [kraken-german-xix-blb-v1]=48caacb4a742b5b5ccd2a92a005a4ca701bdf7cef7859087832b6d726f5f8df4
)

say() { printf '\n== %s ==\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

preflight() {
  say preflight
  echo "host: $(hostname) ($(hostname -f 2>/dev/null || true))  user: $USER"
  [ -d "$SHARE" ] || die "share not mounted: $SHARE"
  mkdir -p "$TMPDIR" ~/atr-cache/wsp ~/Repo
  df -h ~ / 2>/dev/null | sed 1d
  command -v python3.12 >/dev/null || die "python3.12 not found (the trainer's venvs use 3.12.3)"
  command -v git >/dev/null || die "git not found"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    echo "sessions use card 0 only (CUDA_VISIBLE_DEVICES=0); card 1 is the trainer's"
  else
    echo "no nvidia-smi: CPU only, fine for phase 0"
  fi
  ssh -o BatchMode=yes -T git@github.com 2>&1 | head -1 || true
}

checkout() {
  say checkout
  if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only
  else
    git clone "$REMOTE" "$REPO"
  fi
  git -C "$REPO" log --oneline -1
}

venv() {
  say venv
  cd "$REPO"
  [ -x .venv/bin/python ] || python3.12 -m venv .venv
  .venv/bin/pip install --no-cache-dir -q --upgrade pip
  if ! .venv/bin/pip install --no-cache-dir -q -e '.[kraken,dev]'; then
    echo "kraken==7.0.2 did not resolve with pip's torch; pinning torch to the trainer's version"
    TORCH=$(~/Repo/training-atr-models/.venvs/kraken-train/bin/pip show torch 2>/dev/null | awk '/^Version/{print $2}')
    [ -n "$TORCH" ] || die "could not read the trainer's torch version"
    .venv/bin/pip install --no-cache-dir -q "torch==$TORCH" --index-url https://download.pytorch.org/whl/cu128
    .venv/bin/pip install --no-cache-dir -q -e '.[kraken,dev]'
  fi
  .venv/bin/python -c "import kraken, torch, numpy; print('kraken', kraken.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| numpy', numpy.__version__)"
  .venv/bin/python -c "import kraken; assert kraken.__version__.startswith('7.0.'), kraken.__version__" \
    || die "kraken is not 7.0.x — the serving loader (load_any, CoreML) is 7.0.2; fix the pin before going on"
  mkdir -p docs && .venv/bin/pip freeze > docs/ENV_ASTERAIX.txt
}

tests() {
  say tests
  cd "$REPO" && .venv/bin/python -m pytest -q tests
}

weights() {
  say weights
  cd "$REPO"; mkdir -p models/kraken
  {
    echo "# Weight digests on asteraix ($(date -u +%Y-%m-%dT%H:%MZ) UTC)"
    echo
    echo "| id | sha256 | bytes | gateway digest |"
    echo "|---|---|---:|---|"
  } > models/DIGESTS.md
  for id in "${MODELS[@]}"; do
    src="$SHARE/$id/$id.mlmodel"; dst="models/kraken/$id.mlmodel"
    [ -f "$src" ] || { echo "MISSING on share: $src"; echo "| $id | — | — | missing on share |" >> models/DIGESTS.md; continue; }
    [ -f "$dst" ] || cp "$src" "$dst"
    chmod 444 "$dst"
    sha=$(sha256sum "$dst" | cut -d' ' -f1); bytes=$(stat -c %s "$dst")
    if [ -n "${GATEWAY_SHA[$id]:-}" ]; then
      [ "$sha" = "${GATEWAY_SHA[$id]}" ] && gw=match || gw="MISMATCH (gateway ${GATEWAY_SHA[$id]:0:12}…)"
    else gw="not recorded by the gateway"; fi
    echo "| $id | $sha | $bytes | $gw |" >> models/DIGESTS.md
    echo "$id  $sha  $bytes  $gw"
  done
  echo; echo "written models/DIGESTS.md (commit it; models/ itself is git-ignored)"
}

parent() {
  say "parent model (cursive_inzigkofen, $PARENT_DOI)"
  cd "$REPO"
  if [ -f models/kraken/kraken-cursive_inzigkofen.mlmodel ]; then echo "already there"; return; fi
  .venv/bin/python - <<PY || echo "htrmopo could not fetch it; download the .mlmodel of $PARENT_DOI from Zenodo by hand into models/kraken/kraken-cursive_inzigkofen.mlmodel and add its sha256 to models/DIGESTS.md"
import shutil, pathlib
from htrmopo import get_model
p = pathlib.Path(get_model("$PARENT_DOI"))
files = sorted(p.glob("*.mlmodel")) if p.is_dir() else [p]
assert files, f"no .mlmodel under {p}"
dst = pathlib.Path("models/kraken/kraken-cursive_inzigkofen.mlmodel")
shutil.copy2(files[0], dst); dst.chmod(0o444)
print("copied", files[0], "->", dst)
PY
  [ -f models/kraken/kraken-cursive_inzigkofen.mlmodel ] && \
    echo "| kraken-cursive_inzigkofen | $(sha256sum models/kraken/kraken-cursive_inzigkofen.mlmodel | cut -d' ' -f1) | $(stat -c %s models/kraken/kraken-cursive_inzigkofen.mlmodel) | Zenodo $PARENT_DOI |" >> models/DIGESTS.md
}

summary() {
  say summary
  cd "$REPO"
  echo "repo:    $REPO @ $(git log --oneline -1)"
  echo "venv:    $(.venv/bin/python -c 'import kraken,torch;print("kraken",kraken.__version__,"torch",torch.__version__)' 2>/dev/null || echo 'not built')"
  echo "models:  $(ls models/kraken/*.mlmodel 2>/dev/null | wc -l) files under models/kraken/"
  cat models/DIGESTS.md 2>/dev/null || true
  cat <<TXT

next (docs/BRIEF_PHASE0.md §3–§8):
  cd $REPO
  export TMPDIR=~/atr-cache/tmp CUDA_VISIBLE_DEVICES=0
  .venv/bin/python -m pytest -q tests/asteraix        # skips until WSP_MODEL/WSP_PRACTICE are set
  git checkout -b phase0 && git add models/DIGESTS.md docs/ENV_ASTERAIX.txt && git commit -m "asteraix: digests and environment" && git push -u origin phase0
TXT
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(preflight checkout venv tests weights parent summary)
for s in "${steps[@]}"; do "$s"; done
