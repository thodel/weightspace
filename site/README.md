# Public site

The project's public site, in the layout of [dhbern.github.io](https://dhbern.github.io/)
(Quarto, `DHBern/dhbern.github.io`): its colours, navbar, docked sidebar and footer.

- `index.qmd` is the project page. Its front matter follows the dhbern project pages, so it can
  later move to `content/projects/agentic-historian/index.qmd` there.
- `protocol/` is generated from `protocol/docs` (the MkDocs protocol, which stays the source) by
  `scripts/protocol_to_quarto.py`, Quarto's pre-render step. The script fails if a page names an
  internal host or path. The folder is git-ignored.
- `atlas.qmd` presents the activation atlas. `atlas/` is copied from `docs/atlas` (built by
  `wsp.atlas` and `wsp.atlas_explore`) by `scripts/atlas_to_site.py`, the second pre-render step, and
  published unchanged as a Quarto resource: the explorer and the atlas pages keep their own layout.
  Only git-tracked files are copied, and the same check for internal hosts and paths applies. The
  folder is git-ignored.
- The Switzer font is loaded from Fontshare (`custom-header.html`); its licence does not allow the
  font files in a public repository. The DH logo is loaded from dhbern.github.io.

Render locally (Quarto 1.6, Python 3):

```bash
quarto render site
quarto preview site
```

`.github/workflows/site.yml` renders and deploys the site to GitHub Pages on pushes to `main` that touch
`site/`, `protocol/docs/` or `docs/atlas/`, and only while the repository is public.
