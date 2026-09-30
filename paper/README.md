# Paper

[Read the current PDF](https://Victor-Caus.github.io/selective-marl-adaptation/paper.pdf) · [Project page](https://Victor-Caus.github.io/selective-marl-adaptation/)

The current source is **manuscript-v0.2.tex**. It is a standalone English manuscript with embedded TikZ figures, data plots and bibliography. Upload it to Overleaf as `main.tex`, or compile twice with `pdflatex`.

The GitHub Pages workflow compiles this source and deploys the PDF whenever it changes on `main`. The page links to the source revision used for the build. Failed builds leave the last successful publication in place.

## Evidence

`data/liam-three-seeds/` contains the three completed LIAM comparisons, raw session metrics and episode returns, checkpoint hashes and a recovery audit. `data/partial-seed-means.csv` and `figures/` contain derived tables and figures. The original five-seed LIAM criterion was not evaluated. The separate five-seed simpler-control result remains negative.

Historical drafts, the v0.1 builder and operational notes are retained locally. The public manuscript's plots contain the recorded data directly; editing the LaTeX does not retrain or reevaluate models.

## Status

Working manuscript, not peer reviewed. The LIAM implementation is a task port of the released core. Training budget, supervision, partner and recurrent-memory differences are documented in the manuscript. See the main README for the current findings and original-source credits.
