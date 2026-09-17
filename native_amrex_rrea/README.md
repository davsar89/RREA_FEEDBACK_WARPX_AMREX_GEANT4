# Native RREA Coupling

This directory contains the AMReX field/fluid library and the WarpX RZ
overlay used by the production capture path.

`RreaAmrexAdvance` owns the mesh update: low-energy carrier transport by
conservative face number fluxes and the finite-speed Maxwell RZ TM field. The
fluid update and Ampere update consume the same physical face current, while
actual WarpX particle plus fluid charge is checked cell-by-cell against that
current's substep average before the outer step is published. Maxwell TM is
the sole production field path. `RreaWarpXCoupling` binds this state to WarpX particle
containers, diagnostics, and checkpoints.

Runtime geometry, atmosphere, field profile, source, timestep, and diagnostics
are input-driven. What the coupling computes and why is documented in
`docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex`; repository policy is
`CLAUDE.md`.

## Build and checks

Use `scripts/prepare_rrea_warpx_source.py` to create a prepared source tree
from the pinned pristine WarpX source and this overlay. Build products stay outside the
repository.

Building the native targets with `-DRREA_AMREX_BUILD_TESTS=ON` registers every
smoke in this directory as a `ctest` test. Building them is not running them:
run `ctest` and require a clean pass before trusting a change.
