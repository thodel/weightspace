"""Copy the activation atlas (docs/atlas) into the Quarto site as site/atlas/.

docs/atlas stays the source (built by `wsp.atlas` and `wsp.atlas_explore`); this runs as a
Quarto pre-render step, the output folder is git-ignored, and `_quarto.yml` lists it under
`resources`, so Quarto copies it to the site unchanged. The pages are standalone HTML (the
explorer, the per-model viewers and unit cards) and keep their own layout.

The explorer takes the site's style on its own (DH Bern colours in its template); for the site
this step also gives it the site's font (custom-header.html), the light theme only (the site has
no dark theme) and, when it is opened in its own window rather than embedded in atlas.qmd, a bar
in the site's navbar colours that leads back to the site.

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
HEADER = ROOT / "site" / "custom-header.html"
BAR = """<style>.site-bar{display:flex;align-items:center;gap:.6rem;background:#d6002b;color:#fff;
font:500 16px var(--sans);padding:10px 16px;margin:-16px 0 16px;width:100vw;margin-left:calc(50% - 50vw)}
.site-bar a{color:#fff}.site-bar img{height:26px;border-radius:2px;background:#fff}.site-bar .sep{opacity:.7}</style>
<nav class=site-bar><a href="../index.html"><img src="https://dhbern.github.io/logo.png" alt=""></a>
<a href="../index.html">Digital Humanities</a><span class=sep>›</span><a href="../atlas.html">Activation atlas</a></nav>
<script>if(window.self!==window.top)document.querySelector('.site-bar').remove()</script>
"""


def site_explorer(page: str) -> str:
    """The explorer page as the site publishes it: site font, light theme, the bar back to the site."""
    for old, new in (("<html lang=en>", "<html lang=en data-theme=light>"),
                     ("</head>", HEADER.read_text() + "</head>"), ("<body>", "<body>\n" + BAR)):
        if page.count(old) != 1:
            raise SystemExit(f"explorer.html: expected one {old!r}")
        page = page.replace(old, new)
    return page


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
        if rel.as_posix() == "explorer.html":
            dst.write_text(site_explorer(p.read_text()))
        else:
            shutil.copy2(p, dst)
    if hits:
        print("internal names or paths in atlas files (protocol/PUBLISHING.md):", *hits, sep="\n  ", file=sys.stderr)
        return 1
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"atlas: {len(files)} files, {size / 1e6:.1f} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
