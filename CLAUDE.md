# Claude Code

This is an EXPERIMENTAL PHYSICS project (RREA/TGF discharge physics), not a
computer-engineering project. Behave like a physicist, not like a careful
software engineer:

- The deliverable is physical understanding; code and process exist only to
  produce trustworthy numbers.
- One clear path, not option menus. Implement the best physics available,
  check it against physical limits (light cone, relaxation times, known
  regimes, conservation), run, and ask "do the results make sense?" —
  results never need to be byte-identical to anything.
- An unqualified request for a radio plot means the project's standard
  multiband radio figure; an unqualified video request means the standard
  project video/render, never a diagnostic substitute.
- Work directly on the repository's `main` or `master` branch. Do not create
  feature, agent, or task branches unless the user explicitly requests one.
- Judge whether an implementation is more physically accurate by physical
  reasoning alone, never by how little it deviates from a previous run —
  earlier results have no claim to being better. This project explores
  something never really done before; the one strong external anchor is
  the expected RREA multiplication from Coleman & Dwyer (both thresholds,
  276 and 284 kV/m). Published modeling attempts (Dwyer's discharge
  models, Gourbin & Celestin, ...) are peer attempts to compare with, not
  references to match. Everything else is internal consistency.
- Photon transport deliberately excludes coherent Rayleigh scattering.
  Transport is Geant4-informed but the runtime is not branded as option-4 or
  claimed to reproduce its complete process list. State the Rayleigh scope
  when interpreting near-cut angular transport or feedback, and do not add it
  without an explicit user decision.
- This is experimental physics software, not infrastructure-critical code.
  Do not introduce or retain manifest inventories, tracked-file SHA pins,
  report authentication, byte-identity gates, duplicated campaign contracts,
  or historical procedure narratives. None establishes physical correctness.
  Validate physics through primary-source data, governing equations, units,
  conservation, limiting cases, numerical convergence, and direct inspection
  of whether results make physical sense. Git history is the archive. A run
  may record its resolved configuration and Git commit for interpretation,
  but those records are not scientific acceptance gates.
- Concretely: no manifest inventories for project physics tables; no SHA
  values in the canonical defaults; no hashes of reports, plans, scheduler
  scripts, source sets, profiles, figures, or generated schedules; no
  byte-identical regression requirement; no certificate, blessing, or
  report-progression workflow; and no duplicate validator enforcing one
  campaign rule in several layers. Reject malformed data, impossible physical
  values, incompatible binary formats, failed conservation, or failed
  stability/convergence -- not a different path, launcher, rank count,
  machine, or scheduler script. Cross-process expectation contracts are
  ceremony too: one process writing
  down what another ought to produce, then checking it, is a second owner for
  a rule the engine already enforces. Each validation has ONE owner, and the
  engine is that owner wherever it can be -- the checkpoint reader owns
  format, dimensions and state consistency; the driver owns its own arguments;
  a run's recorded configuration is for interpretation, never a gate. Do not
  restate driver defaults, deck contents, or engine invariants in a second
  place so a third place can compare them.
- Treat source lines as a finite budget: changes should be net non-increasing
  unless extra lines are scientifically necessary and no simpler implementation
  exists. Prefer deletion, derivation and consolidation; reject AI-style
  wrappers, defensive scaffolding, configuration mirrors and abstraction bloat.
  This codebase is already large for what it does. Apply the same rule to
  trackers, staged campaigns and procedures: prune complexity rather than
  documenting its branches.
- REQUIREMENT — physics modules: every physical process (a chemistry
  channel, a field update, a deposition, a transport step, a sampler, an
  observable) lives in ONE named, clearly delimited module — a header
  primitive or a named function — with its governing equation or model
  citation stated at the top, reviewable in isolation, and pinned by a
  native smoke or committed test. Physics duplicated across species paths
  or scripts is a defect: one shared primitive, thin adapters, duplication
  budget zero. The architecture guide's physics-module index lists every
  module with its unit and its pin, so any single module can be picked up
  and reviewed alone. Physics that does not meet this bar does not merge.
- Before any cluster build or simulation submission, first complete a clean
  WSL/Linux build of the full production WarpX RZ target with the real RREA
  overlay and run its native physics smokes. Python tests or a native-library
  subset are not substitutes for compiling the production overlay locally.
  This catches ordinary errors cheaply, keeps diagnosis local and simple, and
  avoids wasting the shared compute allocation.
- Local compilation may use at most 12 concurrent CPU jobs/threads. Do not
  derive build parallelism from the logical-CPU count; simultaneous
  multithreading does not justify making the laptop unresponsive. Prefer fewer
  jobs when memory pressure or interactive responsiveness warrants it, and do
  not overlap a heavy local build with a heavy MPI validation run.

CHECKPOINTS MOVE BETWEEN THE HPC CLUSTER AND THIS LAPTOP, AND MUST KEEP DOING
SO — that is how a run continues when the compute allocation runs out, and the
only reason a local build is kept. Pass `--restart-epoch-break` when the rank
count or engine changes; every path may also differ. The last ulp of anything
regenerated through the platform libm can differ too, so compatibility
comparisons of regenerated artifacts must be numeric with a tight tolerance,
never byte-exact. Fetch with
`hpc.py pull-checkpoint`, keep
`core.autocrlf=false`, and if the restart machinery ever refuses a move
because a PATH or a launcher flag changed, that is the machinery being wrong.
The checkpoint reader itself owns format, dimensions and state consistency.
Fetch a whole run for local continuation with `hpc.py pull-restart` — the
checkpoint alone is not enough, the append-only streams come too.

Because checkpoints exist, a running simulation is not a reason to work around
anything. Stop it whenever that is more convenient — to rebuild the engine in
place, migrate a data format, or take the machine back — and restart it from
its checkpoint when done. Do not contort a change, split a build, or defer work
to keep a run alive; a local run is worth far less than the change being made
correctly. Check for a recent `rrea_checkpoint/COMPLETE` before stopping, and
restart with the same or a longer stop time without moving any already-captured
frame.

This file is the repository policy; `AGENTS.md` is a pointer here.
Architecture, rerun, and HPC procedures are indexed by `docs/README.md`;
do not duplicate them here.

NEVER let a turn exceed 8 agents TOTAL: the main loop plus at most 7
sub-agents, counting every nested stage. This one rule is stated in caps
because it is broken by forgetting, not by disagreeing. The reasoning:
fan-out multiplies context, cost and review load faster than it multiplies
insight; a task that seems to need more agents needs a sharper decomposition
instead.
