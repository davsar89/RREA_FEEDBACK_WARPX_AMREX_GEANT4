# Live documentation

This directory keeps only documentation needed to understand, reproduce, and
operate the current implementation. Git history is the archive for completed
reviews, migration plans, benchmark narratives, and superseded runbooks.

The repository root carries `JOURNAL.md`, the chronological record of
campaigns: what was tried, what it measured, and what was rejected. These
documents say what the code is; the journal says how it got there.

## Canonical documents

- `GPU_PORT.md`: experimental hybrid CUDA execution and build, the GPU
  energetic-transport and population machinery and its controls.
- `RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex`: current physics/software
  architecture, including its source-defined data-flow diagram.
- `rrea_pic_explainer.tex`: narrative explainer of how RREA with feedback is
  simulated in the RZ particle-in-cell code and how longitudinal-field and
  optical diagnostics are extracted from it. It explains; the architecture
  guide defines.
- `RREA_RERUN_GUIDE.md`: the current local validation, restart procedure,
  products and all-physics Coleman-Dwyer campaign workflow.
- `OLIVIA_CONNECTION_AND_2FA.md`: the single OLIVIA connection and
  authentication procedure.
- `OLIVIA_COMMAND_WORKFLOW_NOTES.md`: pre-submission checks, CPU build and
  capture, restart, and run and product transfer in both directions.
- `OLIVIA_GPU.md`: GH200 builds and captures, configuration, GPU accounting,
  monitoring, downloads and checkpoint transfer.

LaTeX build intermediates are local artifacts. The two guide PDFs (force-added
past the `*.pdf` ignore) and the README's animation are tracked so the
repository reads without a TeX build; the `.tex` sources stay the truth.

## Literature

`docs/papers/` is the only place locally retained literature files live; some
citations remain DOI-only when a paper is not redistributed. It is tracked
(private remote).
Cite a repo paper when present, otherwise its DOI.

## Source archive

`python MAKE_ARCHIVE.py` writes one gitignored ZIP of the tracked files; its
docstring says what it keeps and skips.

## Vendored dependency: PARMA

`rrea_parma_upstream/` carries the vendored PARMA model; its README states
license, citations and use terms, and `scripts/rrea_parma_source.py` owns the
seed-source physics built on it.

## External dependency: cloudscat

`scripts/rrea_cloud_scattering.py` owns the cloudscat URL, build, cache and
physical self-test. Run it directly to verify the coupling; the architecture
guide owns the optical-transport assumptions and interpretation limits.

## Active physics and interpretation caveats

[RREA_RADIO_LIMITATIONS.md](RREA_RADIO_LIMITATIONS.md) summarizes open physical
consistency issues and interpretation limits for radio and screening results.

C&D rules and statistics live in `scripts/rrea_cd_physics.py` and the report;
`RreaInteractionTables::Load` is the one owner of transport-table semantics.
Model limitations, including Rayleigh exclusion, tracked-brems undercount,
field-taper fringe and observable scope, live in the architecture guide.
