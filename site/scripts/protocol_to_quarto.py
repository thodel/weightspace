"""Copy the MkDocs protocol (protocol/docs) into the Quarto site as site/protocol/.

The MkDocs pages stay the single source; this runs as Quarto's pre-render step and the
output folder is git-ignored. Conversions:

- the first `# heading` becomes the page title (front matter); journal titles come from the
  MkDocs nav, which has the dated short titles;
- `!!! kind "Title"` admonitions become Quarto callouts;
- light/dark figure pairs (`#only-light`, `#only-dark`, `<figure markdown="span">`) keep the
  light figure only, since the site has one theme;
- figures are copied, light versions only;
- `figures/README.md` becomes `figures/index.md`, since Quarto does not render README files.

It also checks every page for internal hostnames and paths (protocol/PUBLISHING.md) and
fails if it finds one, so nothing internal reaches the public site.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "protocol" / "docs"
OUT = ROOT / "site" / "protocol"
MKDOCS = ROOT / "protocol" / "mkdocs.yml"

# protocol/PUBLISHING.md, redaction pass; `atr-cache` is the cache folder's name, not a path
REDACT = re.compile(r"asteraix|dhserver|idhefix|/mnt/|~/|/home/", re.I)
CALLOUT = {"note": "note", "info": "note", "tip": "tip", "warning": "warning", "danger": "important",
           "important": "important", "caution": "caution"}


def nav_titles() -> dict[str, str]:
    """Page path -> title from the `"title": path` lines of the MkDocs nav."""
    out = {}
    for line in MKDOCS.read_text().splitlines():
        m = re.match(r'\s*-\s*"?(.+?)"?:\s*(\S+\.md)\s*$', line)
        if m:
            out[m.group(2)] = m.group(1).replace("\\\"", '"')
    return out


def admonitions(text: str) -> str:
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        m = re.match(r'^(\s*)!!!\s+(\w+)(?:\s+"(.*)")?\s*$', lines[i])
        if not m:
            out.append(lines[i]); i += 1
            continue
        ind, kind, title = m.group(1), m.group(2), m.group(3)
        body, i = [], i + 1
        while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith(ind + "    ")):
            body.append(lines[i][len(ind) + 4:] if lines[i].strip() else "")
            i += 1
        while body and not body[-1].strip():
            body.pop()
        attr = f'.callout-{CALLOUT.get(kind, "note")}' + (f' title="{title}"' if title else "")
        out += [ind + "::: {" + attr + "}", *[(ind + b) if b else "" for b in body], ind + ":::", ""]
    return "\n".join(out)


def figures(text: str) -> str:
    def fig(m):
        imgs = re.findall(r"!\[([^\]]*)\]\(([^)\s]+)\)", m.group(0))
        light = [(alt, src) for alt, src in imgs if "#only-dark" not in src]
        if not light:
            return ""
        alt, src = light[0]
        cap = re.search(r"<figcaption>(.*?)</figcaption>", m.group(0), re.S)
        return f"![{(cap.group(1).strip() if cap else alt)}]({src.replace('#only-light', '')})"

    text = re.sub(r'<figure markdown="span">.*?</figure>', fig, text, flags=re.S)
    text = "\n".join(l for l in text.split("\n") if "#only-dark)" not in l)
    return text.replace("#only-light)", ")")


def convert(path: Path, rel: str, titles: dict[str, str]) -> str:
    text = path.read_text()
    m = re.search(r"^# (.+)$", text, re.M)
    dated = rel.startswith("journal/") and rel != "journal/index.md"
    title = (titles.get(rel) if dated else None) or (m.group(1).strip() if m else path.stem)
    if m:
        text = text[:m.start()] + text[m.end():].lstrip("\n")
    text = figures(admonitions(text)).replace("figures/README.md", "figures/index.md")
    head = "---\ntitle: " + '"' + title.replace('"', '\\"') + '"\n'
    if dated:
        head += f"date: {rel.split('/')[1][:10]}\n"
    return head + "---\n\n" + text


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    titles, hits = nav_titles(), []
    for p in sorted(SRC.rglob("*")):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith("_templates") or p.is_dir():
            continue
        dst = OUT / ("figures/index.md" if rel == "figures/README.md" else rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".md":
            text = convert(p, rel, titles)
            hits += [f"{rel}:{n}: {l.strip()[:100]}" for n, l in enumerate(text.split("\n"), 1) if REDACT.search(l)]
            dst.write_text(text)
        elif p.suffix == ".svg" and not p.stem.endswith("-dark"):
            shutil.copy2(p, dst)
    if hits:
        print("internal names or paths on public pages (protocol/PUBLISHING.md):", *hits, sep="\n  ", file=sys.stderr)
        return 1
    print(f"protocol: {sum(1 for _ in OUT.rglob('*.md'))} pages, {sum(1 for _ in OUT.rglob('*.svg'))} figures -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
