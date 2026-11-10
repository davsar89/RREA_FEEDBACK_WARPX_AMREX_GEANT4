# Agent instructions

This is experimental RREA/TGF discharge physics. Prioritize physical
understanding and trustworthy numbers; choose one best-supported path and
execute it. This file is the project agent policy. Procedures are indexed by
`docs/README.md`; do not duplicate them here.

## Physics and validation

- Validate with governing equations, units, primary-source data, conservation,
  limiting cases (including light cones and relaxation times), numerical
  convergence, and inspection of whether the results make physical sense.
  Previous outputs have no claim to being correct; byte identity is not physics.
- Coleman & Dwyer multiplication, under both 276 and 284 kV/m thresholds, is
  the external anchor. Dwyer discharge models, Gourbin & Celestin and other
  modeling attempts are peers to compare with, not results to match.
- Kinetic photon transport deliberately excludes coherent Rayleigh scattering.
  Describe it as Geant4-informed, not option-4 or a complete Geant4 process
  list. State this scope for near-cut angular transport and feedback; adding
  Rayleigh requires an explicit user decision.
- Every physical process or observable has ONE named, independently reviewable
  function or header primitive, with its governing equation/model citation and
  a native smoke or committed test. Share it across species and scripts through
  thin adapters. The architecture guide's module index records its unit and test.
  Physics that fails this requirement does not merge.
- Each validation has one owner, preferably the engine: the checkpoint reader
  owns format, dimensions and state consistency; the driver owns its arguments.
  Reject malformed data, impossible values, incompatible formats and failed
  conservation/stability/convergence. Do not duplicate defaults, deck contents,
  invariants or cross-process expectation contracts to compare them elsewhere.

## Keep the implementation small

- Work on the existing `main` or `master`; create no branches/worktrees unless
  requested. Preserve unrelated user changes and act autonomously within scope.
- Prefer deletion, derivation and consolidation. Changes should be net
  non-increasing in source lines unless scientifically necessary with no simpler
  implementation. Avoid speculative wrappers, defensive scaffolding,
  configuration mirrors and duplicate validators; prune trackers and procedures.
- No manifest inventories, file-hash pins, report authentication, certificates,
  report approval/progression workflows,
  or byte-identical regression gates. Git history is the archive. A run may record
  its resolved configuration and commit for interpretation, never as acceptance.
- Use proportionate tests of physical and operational behavior, not identities
  of paths, machines, launchers, rank counts or serialized bytes.
- At most 8 agents total, including the main agent and every nested sub-agent.

## Products

- An unqualified radio plot means the standard multiband figure; a video means
  the standard project render. Do not substitute diagnostics.
- Video capture cadence is 1 microsecond. Receive user approval before changing
  it; storage or performance optimization does not authorize a change.
- Download products to repo-root `eNNN_radio/` and `eNNN_videos/` (gitignored;
  NNN is peak E0 in kV/m). Only a different E0 creates another folder; encode
  run/condition differences in filenames.
- Optical plots use blue for 337 nm and red for 777 nm, with line styles and
  shades distinguishing other variables. Cloudscat raw timelines are photons
  m^-2 s^-1 and images photons m^-2 sr^-1, both per source photon; preserve the
  detector-area factor when converting them to physical observables.
- At every performance-round close, refresh Table 1 and the round table in the
  architecture TeX guide, rebuild and visually inspect the PDF. Show the latest
  retained measurements; label unremeasured configurations historical and never
  present mixed code versions as current-code scaling.

## Compute and OLIVIA

- Before any OLIVIA build or simulation submission, complete a clean local
  WSL/Linux build of the full production WarpX RZ target with the real RREA
  overlay, then run its native physics smokes. Python tests or a native-library
  subset do not substitute for the production build.
- Local compilation uses at most 12 CPU jobs/threads, fewer when memory or
  responsiveness warrants. Do not derive this from logical CPU counts or
  overlap a heavy build with heavy MPI validation.
- Use OLIVIA's `devel` QoS for builds, tests, debug runs and benchmarks shorter
  than two hours; check current scheduler limits before submission.
- Size OLIVIA RAM requests from measured peak usage plus modest headroom.
  Check memory-based CPU billing so excess requested RAM does not waste quota.
- Check `uname -r` before OLIVIA access and use `scripts/olivia.py`, which picks
  the route. Native Linux uses the `ssh olivia` ControlMaster; reopen an expired
  master with `bash -ic 'ssh olivia true'`, never ask the user to log in.
  Windows/WSL uses the mux socket from `scripts/olivia_open_socket.sh`.
  Connection details belong in `docs/OLIVIA_CONNECTION_AND_2FA.md`.

## Checkpoints and continuation

- Checkpoints must move between OLIVIA and the laptop. Use
  `--restart-epoch-break` when rank count or engine changes; paths may all differ.
  Compare regenerated artifacts numerically with tight tolerances, allowing
  platform-libm rounding, never byte-exactly. Keep `core.autocrlf=false`;
  rejecting a restart because a path or launcher flag changed is a bug.
- Fetch a whole continuation with `olivia.py pull-restart`: checkpoints and
  append-only streams are both required.
- A running simulation need not delay a rebuild, format migration or resource
  recovery. Verify a recent `rrea_checkpoint/COMPLETE` before stopping, then
  restart with the same or longer stop time without moving captured frames.
  Do not add workarounds merely to keep a run alive.
