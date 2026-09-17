# Live documentation

This directory keeps only documentation needed to understand, reproduce, and
operate the current implementation. Git history is the archive for completed
reviews, migration plans, benchmark narratives, and superseded runbooks.

## Canonical documents

- `GPU_PORT.md`: experimental hybrid CUDA execution, build/runtime requirements,
  measured validation scope and the remaining CPU transport work.

- `RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex`: current physics/software
  architecture, including its source-defined data-flow diagram.
- `rrea_pic_explainer.tex`: narrative explainer of how RREA with feedback is
  simulated in the RZ particle-in-cell code and how longitudinal-field and
  optical diagnostics are extracted from it. It explains; the architecture
  guide defines.
- `RREA_RERUN_GUIDE.md`: the current local validation, restart procedure,
  OLIVIA capture submission, and all-physics Coleman-Dwyer campaign
  workflow.
- `OLIVIA_CONNECTION_AND_2FA.md`: the single OLIVIA connection and
  authentication procedure.
- `OLIVIA_COMMAND_WORKFLOW_NOTES.md`: source transfer, Slurm build/run,
  monitoring, quota, retrieval, and the ordered capture-to-deliverable
  procedure (reduce on the host, render, pull, draw).

LaTeX build intermediates and rendered PDFs are local artifacts, not additional
sources of truth.

## Literature

`docs/papers/` is the only place locally retained literature files live; some
citations remain DOI-only when a paper is not redistributed. It is tracked
(private remote), via a `!docs/papers/` exception to the blanket
`*.pdf` ignore, and `*.pdf binary` in `.gitattributes` keeps the bytes exact.
Compiled PDFs of the `.tex` documents above stay untracked: they are build
products, not sources. Cite a repo paper when present, otherwise its DOI.

## Source archive

`python MAKE_ARCHIVE.py --review` writes one gitignored ZIP containing the
project's tracked regular files, including docs, tests, packed schema-6 tables
and validation holdouts. WarpX/AMReX/Geant4 source trees remain external; the ZIP
contains `.gitmodules` with their fork URLs, while a Git
clone also checks out their exact gitlinks. Captures, renders, builds and
credentials stay outside it.

## Vendored dependency: PARMA

`rrea_parma_upstream/` carries the PARMA cosmic-ray model code and data
(license and citations in that directory; non-commercial JAEA scope).
`scripts/rrea_parma_source.py` is the single owner of the seed-source
physics built on it, of the thin evaluator's gfortran build, and of the
standalone audit mode; run it directly to print the flux/bath profile.

## External dependency: cloudscat

`scripts/rrea_cloud_scattering.py` owns the cloudscat URL, WSL build, cache and
physical self-test. Run it directly to verify the coupling; the architecture
guide owns the optical-transport assumptions and interpretation limits.

## Active physics and interpretation caveats

[RREA_RADIO_LIMITATIONS.md](RREA_RADIO_LIMITATIONS.md) summarizes open physical
consistency issues and interpretation limits for radio and screening results.

C&D rules and statistics live in `scripts/rrea_cd_physics.py` and the report;
`RreaInteractionTables::Load` is the one owner of transport-table semantics.
Model limitations, including Rayleigh exclusion, tracked-brems undercount,
field-taper fringe and observable scope, live in the architecture guide.

## Experimental GPU backend

See [`GPU_PORT.md`](GPU_PORT.md) for the CUDA build and acceptance path and
[`GPU_KINETICS.md`](GPU_KINETICS.md) for energetic transport, secondary/population
execution, ownership and remaining host work. Native CUDA validation is still
required before production use.
