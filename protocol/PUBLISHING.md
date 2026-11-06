# Publishing checklist (GitHub Pages)

Nothing is published yet. The workflow `.github/workflows/protocol-pages.yml`
runs only when started by hand (`workflow_dispatch`); no push deploys.

## Before the first publication

- [ ] **Where it lives.** `thodel/weightspace` is private. Pages from a private
      repo needs a paid plan, and the site is then public anyway. Either make
      the repo public, or copy `protocol/` (plus the workflow) into a separate
      public repo, e.g. `thodel/weightspace-protocol`.
- [ ] **Redaction pass** over every page: no internal hostnames, server or
      share paths, user names, job ids that reveal infrastructure, personal
      data or credentials. `grep -rniE "asteraix|dhserver|idhefix|tei|/mnt/|~/|atr-cache" protocol/docs`
      should return nothing.
- [ ] **Data and model references**: name public datasets and models by their
      public identifiers (HF / Zenodo DOIs); state which in-house models and
      data are not public.
- [ ] **Dataset defect** (journal, phase 0): confirm the dataset owner was told,
      before it is published.
- [ ] **Licence**: confirm CC BY 4.0 for text (the `copyright` line in
      `mkdocs.yml`), and the code licence if the repo goes public.
- [ ] **Authorship / acknowledgements**: names, affiliation, funding, and how
      AI assistance (the programmer models) is credited.
- [ ] Set `site_url` and `repo_url` in `mkdocs.yml`.
- [ ] GitHub → Settings → Pages → Source: **GitHub Actions**.
- [ ] Run the workflow (Actions → *Protocol site* → Run workflow), check the
      page, then optionally add a `push` trigger on `main` for `protocol/**`.

## Optional

- A DOI for a frozen version (Zenodo GitHub integration on a release tag).
