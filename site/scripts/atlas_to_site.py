"""Copy the activation atlas (docs/atlas) into the Quarto site as site/atlas/.

docs/atlas stays the source (built by `wsp.atlas` and `wsp.atlas_explore`); this runs as a
Quarto pre-render step, the output folder is git-ignored, and `_quarto.yml` lists it under
`resources`, so Quarto copies it to the site unchanged. The pages are standalone HTML (the
explorer, the per-model viewers and unit cards) and keep their own layout.

Only files tracked by git are copied: docs/atlas/img/ is git-ignored and holds more line images
locally than the twelve practice lines committed for the explorer and viewers.

Like protocol_to_quarto.py, it fails if a text file names an internal host or path
(protocol/PUBLISHING.md).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "atlas"
OUT = ROOT / "site" / "atlas"
REDACT = re.compile(r"asteraix|dhserver|idhefix|/mnt/|~/|/home/", re.I)
TEXT = {".html", ".js", ".json", ".svg", ".md"}


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--", str(SRC)], cwd=ROOT, capture_output=True, check=True)
    return [ROOT / p for p in out.stdout.decode().split("\0") if p]


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    files, hits = tracked(), []
    if not any(p.name == "explorer.html" for p in files):
        print(f"no tracked explorer.html under {SRC.relative_to(ROOT)}", file=sys.stderr)
        return 1
    for p in files:
        rel = p.relative_to(SRC)
        if p.suffix in TEXT:
            hits += [f"{rel}:{n}" for n, l in enumerate(p.read_text().split("\n"), 1) if REDACT.search(l)]
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    if hits:
        print("internal names or paths in atlas files (protocol/PUBLISHING.md):", *hits, sep="\n  ", file=sys.stderr)
        return 1
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"atlas: {len(files)} files, {size / 1e6:.1f} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
