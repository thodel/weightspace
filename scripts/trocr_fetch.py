#!/usr/bin/env python3
"""Fetch the TrOCR family of docs/MODELS_TROCR.md for reading (stage 0 of docs/epics/EPIC_TROCR_TRACK.md).

Run on asteraix, never on h1 (5 GB free there). Read-only on the hub: nothing is uploaded. Uses the credentials
huggingface_hub finds itself (never printed here). Private repositories need --private-ok on purpose.

    python scripts/trocr_fetch.py --dry-run                 # sizes, private flag, no download
    python scripts/trocr_fetch.py --set core                # the Bern chain
    python scripts/trocr_fetch.py --set core --private-ok   # plus the two private repositories
    python scripts/trocr_fetch.py --set all --private-ok

Writes ``~/atr-cache/trocr/<owner>__<name>/`` and ``docs/trocr/DIGESTS.json`` (repo, revision, file sizes, sha256 of
model.safetensors and of the tokenizer files). Weights are never committed; the digests are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CORE = [
    "microsoft/trocr-base-handwritten",
    "dh-unibe/trocr-kurrent",
    "dh-unibe/trocr-kurrent-XVI-XVII",
    "dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2",
    "dh-unibe/trocr-german-kurrent-xvi-xix-v2",        # private
    "dh-unibe/trocr-german-kurrent-xvi-xix-full-v1",   # private
    "dh-unibe/trocr-medieval-escriptmask",
]
EXTRA = [
    "microsoft/trocr-base-stage1",
    "medieval-data/trocr-medieval-base",
    "LaMOP/TrOCR_Manicule_2026_Latin_Medieval",
    "Riksarkivet/trocr-base-handwritten-hist-swe-2",
    "kazars24/trocr-base-handwritten-ru",
    "fgho/trocr-hanseXVI-kurrent",
]
LARGE = [
    "microsoft/trocr-large-handwritten",
    "dh-unibe/trocr-essoins-middle-latin",
    "magistermilitum/tridis_HTR",
    "LaMOP/TrOCR_Large_Manicule_2026_Latin_Medieval",
]
SETS = {"core": CORE, "extra": CORE + EXTRA, "all": CORE + EXTRA + LARGE}
PATTERNS = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer*.json", "vocab.json",
            "merges.txt", "special_tokens_map.json", "model.safetensors", "training_summary.json", "metadata.json",
            "README.md", "training_args.bin"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), default="core")
    ap.add_argument("--repo", action="append", help="extra repo id (repeatable)")
    ap.add_argument("--dest", default=str(Path.home() / "atr-cache" / "trocr"))
    ap.add_argument("--digests", default="docs/trocr/DIGESTS.json")
    ap.add_argument("--private-ok", action="store_true", help="allow private repositories")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download  # lazy: not installed on h1

    api = HfApi()
    repos = SETS[a.set] + (a.repo or [])
    out: dict[str, dict] = {}
    dig = Path(a.digests)
    if dig.exists():
        out = json.loads(dig.read_text())
    for rid in repos:
        info = api.model_info(rid, files_metadata=True)
        files = {s.rfilename: s.size for s in (info.siblings or [])}
        w = files.get("model.safetensors")
        print(f"{rid}: private={info.private} revision={info.sha[:12]} model.safetensors={w and round(w / 1e9, 2)} GB")
        if a.dry_run:
            continue
        if info.private and not a.private_ok:
            print("  skipped: private (pass --private-ok)")
            continue
        dest = Path(a.dest) / rid.replace("/", "__")
        snapshot_download(rid, revision=info.sha, local_dir=str(dest), allow_patterns=PATTERNS)
        rec = {"revision": info.sha, "private": bool(info.private), "files": {p.name: p.stat().st_size for p in sorted(dest.glob("*")) if p.is_file()}}
        for name in ("model.safetensors", "vocab.json", "merges.txt", "tokenizer.json", "preprocessor_config.json", "config.json"):
            if (dest / name).exists():
                rec.setdefault("sha256", {})[name] = sha256(dest / name)
        out[rid] = rec
        dig.parent.mkdir(parents=True, exist_ok=True)
        dig.write_text(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
