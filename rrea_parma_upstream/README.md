# Vendored PARMA model subset

This directory contains the subset of the PARMA atmospheric cosmic-ray model
used by the RREA seed-source module:

- `subroutines.f90`: PARMA spectral and angular routines;
- `input/`: the data tables those routines read;
- `LICENSE`: upstream use and attribution terms.

It is not a standalone copy of the upstream calculator. Build files, example
drivers, plots, Docker files, and upstream regression tests are intentionally
not vendored, so commands from the upstream project README do not apply here.

`scripts/parma_electron_query.f90` is the thin project adapter, and
`scripts/rrea_parma_source.py` owns compilation, numerical integration,
electron/positron sampling, source normalization, and the production audit.
On a fresh PARMA-source run it builds the adapter with `gfortran`, queries this
vendored model, and stores `parma_source_table.txt` in the run directory.
Restart schedule generation reads that stored table and does not require a
compiler or another PARMA query.

The production integration uses electron ID 31, positron ID 32, and PARMA
angular class 5. Its fixed solar, atmosphere, and geometry assumptions and its
configurable geographic inputs are documented with the governing equations in
`scripts/rrea_parma_source.py`; run defaults live only in
`config/rrea_defaults.json`.

## Attribution and use terms

The vendored model originates from
[COSMIC_RAY_FLUXES](https://github.com/davsar89/COSMIC_RAY_FLUXES) and is based
on:

1. T. Sato, “Analytical model for estimating terrestrial cosmic ray fluxes
   nearly anytime and anywhere in the world: Extension of PARMA/EXPACS,”
   *PLOS ONE* 10(12), e0144679 (2015).
2. T. Sato, “Analytical model for estimating the zenith angle dependence of
   terrestrial cosmic ray fluxes,” *PLOS ONE* 11(8), e0160390 (2016).

The original materials require citation for non-commercial use and prior JAEA
agreement for commercial use. This subset has no general open-source license
grant; consult `LICENSE` before redistribution or commercial use.
