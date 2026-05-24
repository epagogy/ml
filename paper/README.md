# paper/ — JOSS submission for the ml package

This directory contains the JOSS (Journal of Open Source Software) paper for the `ml` package.

## Status

**STUB** — authoring deferred per the §5 checkpoint resolved 2026-05-18. See `epagogy/science/ml/packages/ml/JOSS_PLAN.md` for the full submission strategy.

## Files

- `paper.md` — JOSS paper (pandoc markdown, ~1000 words target)
- `paper.bib` — BibTeX references
- `README.md` — this file

## Building the PDF (when ready)

JOSS provides a Docker-based PDF builder. Typically wired as a GitHub Action: `.github/workflows/draft-pdf.yml` — to be added when authoring begins.

Local build:
```bash
docker run --rm \
    --volume $PWD/paper:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
```

## Submission procedure

1. Author `paper.md` to JOSS spec
2. Populate `paper.bib`
3. Wire up `.github/workflows/draft-pdf.yml` for CI PDF builds
4. Open submission PR at https://github.com/openjournals/joss-reviews
5. Reviewer addresses comments
6. Acceptance + DOI minted
7. Update `CITATION.cff` in `packages/public/ml/` with the JOSS DOI

## JOSS reference

https://joss.readthedocs.io/en/latest/submitting.html
