# Publishing checklist (GitHub Pages)

Nothing is published yet. The public site is the Quarto site in `site/` (dhbern layout, see
`site/README.md`); it includes these protocol pages, converted at render time. The workflow
`.github/workflows/site.yml` deploys it on pushes to `main`, but only while the repository is
public. The MkDocs build here remains the strict check of the protocol pages.

## Before the first publication

- [ ] **Where it lives.** `thodel/weightspace` is private. Pages from a private
      repo needs a paid plan, and the site is then public anyway. Either make
      the repo public, or copy `protocol/` (plus the workflow) into a separate
      public repo, e.g. `thodel/weightspace-protocol`.
- [x] **Redaction pass** over the site pages: `site/scripts/protocol_to_quarto.py` fails the render
      if a page names an internal host (`asteraix`, `dhserver`, `idhefix`) or path (`/mnt/`, `~/`,
      `/home/`).
- [ ] **Redaction of the repository**, if the repository itself goes public: on `main` (18 Nov),
      748 files contain `/home/tobias` paths, 84 the research-share path `/mnt/wbkolleg…`, 57 the
      server name. No credentials were found in the history.
- [ ] **Data and model references**: name public datasets and models by their
      public identifiers (HF / Zenodo DOIs); state which in-house models and
      data are not public.
- [ ] **Dataset defect** (journal, phase 0): confirm the dataset owner was told,
      before it is published.
- [x] **Licence**: text, figures and run records CC BY 4.0 (`LICENSE-CC-BY-4.0`), code MIT
      (`LICENSE`); `data/` and third-party models keep their owners' licences (README, *Licence*).
- [ ] **Authorship / acknowledgements**: names, affiliation, funding, and how
      AI assistance (the programmer models) is credited.
- [x] `site-url` and `repo-url` are set in `site/_quarto.yml` (`thodel.github.io/weightspace`).
- [ ] GitHub → Settings → Pages → Source: **GitHub Actions**.
- [ ] Run the workflow (Actions → *Site* → Run workflow) and check the page.

## Optional

- A DOI for a frozen version (Zenodo GitHub integration on a release tag).
