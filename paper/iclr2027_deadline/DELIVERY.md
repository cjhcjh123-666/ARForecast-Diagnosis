# Deadline paper bundle

This directory is independent from `paper/iclr2027/` and contains the current
anonymous deadline draft.

- `main.tex`, `refs.bib`, styles, and `figs/`: main-paper source bundle.
- `main.pdf`: compiled main-paper draft.
- `verified_diagnostics.tex`: evidence-locked Level-1, Level-2 coverage, and
  IRTS diagnostic section included by `main.tex`.
- `supplement.tex` and `supplement.pdf`: anonymous supplementary draft.

Compile from this directory with:

```bash
/public/duyinglong/miniconda3/envs/latex/bin/tectonic main.tex --keep-logs --keep-intermediates
/public/duyinglong/miniconda3/envs/latex/bin/tectonic supplement.tex --keep-logs --keep-intermediates
```

Both PDFs compile successfully with Tectonic 0.17.0. The main bibliography has
one pre-existing metadata warning (`ansari2024chronos` has an empty
`booktitle`); there are no undefined citations, missing figures, or overfull
boxes in the latest compile. Quantitative provenance is maintained in
`docs/deadline_execution/claim_evidence_matrix.csv`.

The manuscript deliberately withholds legacy Level-2 and native-QA results
that failed the protocol audit. Corrected partial coverage is labelled as such
and must not be rewritten as full ten-model evidence.
