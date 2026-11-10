# Journal

Chronological record of campaigns. What the code *is* lives in `docs/`; this
file records what was tried, what it measured, and what was rejected, so a
later session does not repeat a dead end. Newest first.

## 2026-10-28: Energetic-electron first-passage map

The history-enabled production build passed 29 native and 25 delivery tests;
splitting and a two-to-four-rank restart retained 10.75 physical electrons.
OLIVIA build 2665210 and capture 2665235 replaced the single-map build 2663525
and capture 2663574 (cancelled before starting). The E125 pulse-plus-PARMA
repeat targets 1 ms in `runs/scratch_e125_pulse_parma_creation_1ms_20261028`.
Observable and usage: architecture and rerun guides.

## 2026-10-28: Radio reduction

OLIVIA benchmark 2660207 measured a 5x radio speedup with unchanged sampling;
48 physics/delivery tests passed. Random-read advice remained filesystem-latency
bound and was rejected. Benchmark source and raw timings:
`warpx_rrea_smoke/tmp/radio_production_benchmark_20261028.{py,json}`.
Measurements and scope are in the architecture guide's performance tables.

The direct C++/OpenMP port failed the 50% time-saving requirement; Numba stays.
Screen 2664522 ran on `devel`; source and all paired timings remain in
`warpx_rrea_smoke/tmp/radio_cpp_20261028/` (`screening.json`). No production
C++ implementation or dependency was added; R9 records the comparison.

## 2026-10-27: Molecular optical assessment and cutoff study

Findings and retained physics are in the architecture guide's optical and
Coleman-Dwyer sections. In OLIVIA's `warpx_rrea_smoke`, the raw charged
references and study bundles are in `molecular_20261027/`, and captures in
`runs/scratch_molecular_unit_*`. Local MCAIR diagnostic scripts/results are in
`.codex_tmp/molecular/`; their source revision was not recorded, so those
comparisons remain exploratory. No native correction was adopted from them.

## 2026-09-28: Bloat, staleness and second-owner cleanup

Removed dead code and restated facts across docs, launchers and engine (net
about 4.1k lines, 2.4k of them unread generator contracts in
`transport_physics.json`, plus 11 unread holdout files). Interface change: the OLIVIA capture sbatch,
`submit_profiled_video.sh` and the C&D planner now pass driver flags verbatim
instead of ~45 `PROFILED_*` variables; restart is `--restart-checkpoint auto`
with the same `RUN_TAG`, and the credential helpers read only the private
state directory. Engine: dead direct negative-ion source channel, one
reduced-CSV column list, RZ-only build. Checked locally: all 28 RREA smokes and
128 Python tests pass; a 40-step E120 capture has the same reduced-CSV header
and bit-identical values as the pre-cleanup engine. A checkpoint written by the
old engine restarts under the new one at 2 ranks (bit-identical endpoint) and
at 3 ranks with an epoch break (new realization, same-scale energy residual,
all conservation gates passed). Observed, not fixed:
`em_charge_continuity_signed_accum_e` is a process-lifetime sum absent from the
checkpoint, so after a restart it covers only the post-restart steps.
Fixed the same day: it now travels in `maxwell_tm_state.bin` (a trailing
double; older checkpoints load and restart it at 0). `global_jz_fluid_A_m` and
the fluid bins of `current_profile.csv` switched from the diagnostic cell
sigma*Ez to the step-averaged fluid face current Maxwell receives; fluid plus
net-chord now equals `maxwell_jz_moment_A_m` to 1e-14, and the radio figure's
bin gate no longer refuses short captures (residual 0.7%, kinetic sampling).
Captures continued across this change mix the two fluid definitions at the
restart. An empty species now renders as an empty panel.

## 2026-09-28: Storage cleanup under PRDR

User-authorized deletions in `warpx_rrea_smoke` (118 paths, about 190 GiB): 98
`runs/profiled_video_schema6_*` archive copies from August to 3 September
(superseded development engines), the August scratch development runs
(`scratch_maxwell_e125_200us*`, `scratch_mgs*_probe`, `scratch_e125_*`,
`scratch_cd_e90_r0`, `e125_v4_handoff_test`) and the pre-PARMA
`scratch_e130_200us`; the second copy of the E130 PARMA run
(`profiled_video_schema6_2129357` and its `_latest` link; the working copy
`scratch_parma_e130_200us` stays); and the raw benchmark captures
`runs/cpu_gpu_20260921`, `gpu_route_candidate`, `cpu_optimization_20260921`
and `cpu_reference_20260919` (their timings live in the guide and this file).
Kept: the E120 PARMA and pulse runs, the E130 PARMA run, both C&D ensembles,
the referenced GPU benchmark runs, builds, sources, the products venv and logs.

## 2026-09-26: Radio review against the E120 capture (main 8850c69)

Kernel signs checked by hand; recorder samples lie exactly on the surface. E120
at the 100 km ground receiver: E_y, B_x, B_z at round-off; E_theta against
cB_phi correlation 0.9994; late static field -0.143 V/m against -0.148 V/m from
the Ampere-sourced moment M(t); 3-12 kHz surface/dipole amplitude 0.97-0.99.
98% of peak M (139 kA km) is fluid conduction with rms_r 1.88 km, so above the
26 kHz radial corner the ratio falls (0.47 at 25-50 kHz, 0.28 at 100-200 kHz)
and the LF dipole overlay (14.7 against 6.8 V/m) is now drawn only below it.
LF content depends on unconverged source structure (macro count, dt).
Negligible: escaped charge at most 0.5 mC against |p| = 1.3e4 C m; Gauss
residual 2e-12 at the M peak; stride-4 aliasing at most 4e-6 of band energy.
Aircraft azimuth check (N 256 vs 512, job 2364493, 11 billing h): E 0.63%,
B 1.8%, each half the field's content above the 1.23 MHz trapezoid limit, where
the field is numerical anyway; N = 256 stands. The delivered E120 figure
predated 91ce0a8 on OLIVIA (Hann-doubled spectrum, zero-phase filters) and was
redrawn: LF peak 5.8 V/m, VLF 2.7 against dipole 2.8 V/m, nadir record-end
level -512 V/m still moving -38 V/m over the last 51 us.

## 2026-09-25: Coleman-Dwyer check at bounded population (main 51c8c72)

Production GH200 path (host kinetics, GPU fluid/Maxwell), 100k C&D seeds,
10 us, dt 2.5 ns, engineering realizations 0-2. Unit weight costs scale with
M: 90 kV/m took ~1 GPU-h per realization, 105 kV/m was cancelled at 3.5 h
(extrapolated ~75 h), 119 kV/m cannot fit one GH200. With the production
controller at 100k electron macros every case took 23-29 min (~4 GPU-h for
nine). Bridge at 90 kV/m: M = 11.189 (unit, CI +/-0.31%) vs 11.166
(resampled, +/-4.5%), -0.21%: no controller bias on M. Thinning noise is ~1.8%
per realization, so three realizations miss the 3% CI gate; it changes no
class. Results (ratio to the 276 curve, 95% CI; the 284 class):
90 kV/m M = 11.17, 0.894 [0.855, 0.936] consistent (284 high);
105 kV/m M = 280.5, 0.865 [0.831, 0.900] consistent (284 consistent);
119 kV/m M = 9146, 0.857 [0.767, 0.959] consistent (284 consistent).
In avalanche lengths the runs sit 4.4, 2.5 and 1.7% below the 276 curve,
between 276 and 284. Core field perturbation <= 2.2e-4. Timestep convergence
(dt 1.25/0.625 ns) not run.

## 2026-09-23: Staleness and second-owner audit

Docs, docstrings and code checked against the tree; about -1,500 lines. Bugs
found: `olivia.py build` ran its smokes on the `sync` copy's tables, not the
built commit's; three tiled diagnostic loops summed `validbox()` per tile (right
only while the 2D tile size never splits a box). Second owners removed: the C&D
planner's seed, ceiling and radius exports (the driver derives them), the
engine's `rrea.transport_model` key (the table owns the model), sbatch re-checks
of driver rules. Reduced CSV loses `ionization_event_count` (a sum of densities,
unread), so a running capture continues only with `--restart-epoch-break`. A
40-step 2-rank capture and its restart from step 20 ran clean (continuity 1e-16).

## 2026-09-23 -- Engine bloat pass (commits after aaf5695)

About -4,700 native lines in 30 commits; physics unchanged (a 20-step 2-rank
run before and after matches in every physics column, EM monitors to
round-off). Checkpoint break: rank-state v9 (38 unread ledger fields, retired
counters and the stored Maxwell time dropped), so a pre-v9 capture continues
only with the engine that wrote it. Output changes: reduced CSV loses its
constant, derived and zero columns (rng_seed, table class, N_inst/N_cumulative,
hook counts, min_tau_m, orphan weight); rrea_video loses
field_direction_moments.csv and unread frame/moment columns. Deck keys that
restated the tables (photon cutoff, hard-Moller threshold) or another key are
gone. Found, not changed: at 100 MeV the median multiple-scattering 1-cos
jumps 47% (1.01e-7 GS to 1.48e-7 WentzelVI at n=10) across the model switch;
few runaway electrons reach it.

## 2026-09-23 -- Review fixes (commits 3589364..a982ed0)

A full review found result-changing defects; all are fixed on gpu-cuda, net
about -2,900 lines. Physics changes that make E120/E130 non-comparable with new
runs: PARMA stream is the CSDA birth rate phi*S*chi (4.6e13 /s over 6-16 km,
was 1.7e11 /s from phi/H); thinning spread by capacity (was axis-first);
three-body attachment below 1 Td from Kossyi(Te) (was 9x high at 0.1 Td);
closure capped at 60 Td; drag frozen at the substep midpoint; one charged and
one photon history (newborn exit/demotion order fixed). Rejected: moving the
top boundary above 16 km. The curl-free taper's fringe field then exceeds the
thin air's threshold (1.03-1.34x at 16.5-18 km, thousands of pockets), so only
the bottom moved (9 -> 6 km). Not yet done: a CUDA build and an OLIVIA C&D
engineering cohort at 90/105/119 kV/m to confirm the merge leaves M unchanged.

## 2026-09-21 -- E120 final products at 200 us

Final video and multiband radio figure: `results/e120_gh200_final_200us_20260921/`;
remote `runs/scratch_e120_gh200/products/final_200us`. Video: 1681 frames through
199.92 us (the last scheduled frame), 70.042 s, 5868x2880, 24 fps; full decode
and middle/final-frame visual checks passed. All three observer traces have
finite physical fields and increasing times; optical normalization, arrival
and scaling checks passed. Radio figure visually checked; short-record VLF
and cloud-delay qualifications remain on the figure.
Products cost 2.085 GPU h: radio 2297775 (stopped after the figure to avoid a
duplicate render), video 2297868, caption correction 2298019.
The rerun guide records the standard commands without another wrapper;
ARM CloudScat selection is automatic and sub-unity ratios no longer print as
zero. Validation: 53 focused Python checks and Ruff E9/F passed.

## 2026-09-20 -- Documentation and test audit

Reviewed the 18 tracked Markdown/TeX files and 248 Python docstrings, with
implementation/reference checks and native-test review. Corrected CUDA status,
arena ownership, Gauss/injection and boundary caveats, Huygens versus fallback
products, and the index's 7300-kV avalanche-integral denominator. Removed
unreferenced phase instrumentation and duplicate GPU campaign narrative.
Replaced self-derived geometry/closure assertions and local-capture-dependent
optical checks; added band-volume, codec-layout/corruption, source-coupling
units and missing-smoke-output coverage. Missing tracked tables now fail tests.
Validation: 138 WSL Python tests, 28 existing native smokes, Ruff E9/F,
and both TeX builds; edited pages visually checked. Numerical runtime unchanged.

## 2026-09-20 -- Storage cleanup and E120 continuation

User-authorized deletions under PRDR: old `warpx_rrea_smoke/gpu_cuda` and
`opt` builds, `dsarria/backups_20260714`, the two-stage CORSIKA
`output/SUPERSEDED_pre_gulf_14km_20260915` archive, and RREA runs
`branches_b950577_chk016160` and `scratch_maxwell_e125_200us_03bb285`.
Old CPU replays may require rebuilding. Pinhole cleanup removed 16.14 GiB of
superseded results and old event files covered by retained response databases;
newer campaigns and shared libraries remain. Cache cleanup saved 8.80 GiB.

Job 2295447 completed E120 from chk043080 (107.7 us) to step 80000 / 200 us,
one GH200 + 48 host transport threads, retained R6 configuration and unchanged
epoch/capture schedule. Full local production build and 28 native checks passed.
Complete chk080000 and native exit 0 confirmed September 21. Final continuity
residual 1.18e-16, Gauss residual 1.14e-11; recent diagnostics finite.
Capture: `$PRDR/dsarria/warpx_rrea_smoke/runs/scratch_e120_gh200/capture`.
Command and launch script: `tmp/olivia_gpu_rounds_20260919/e120_200us_command.json`
and `e120_to_200us.sh`, mirrored in remote `runs/gpu_rounds_20260919`.

Attempt 2295170 failed at 108 us with ENOSPC writing chk043200, costing
0.2033 GPU h. Diagnostics remained finite (continuity 1.91e-16, Gauss 2.55e-8,
energy residual 6.15e-11 J). Another GPU node reproduced the work-filesystem
write failure despite ample reported space; PRDR writes succeeded. Cause
remains unresolved. The incomplete checkpoint was excluded on transfer.
The obsolete work copy at `/cluster/work/projects/nn9526k/dsarria/e120_200us_20260920`
can be removed once its replacement PRDR checkpoints are confirmed.

Continuation cost: 19.9736 GPU h (71905 s), within the initial 18--20 h estimate.
Use `sacct -X -j 2295170,2295442,2295443,2295445,2295447 --format=JobID,State,ElapsedRaw,AllocTRES`;
GPU h = allocated GPUs times ElapsedRaw / 3600. Storage probes cost 0.00083 GPU h.
No further optimization round started.

## 2026-09-20 -- E120 preview products complete

Job 2293277 continued chk042480 (106.2 us) to clean stop chk042842 (107.105 us),
one GH200/host48, preserving the 200 us capture schedule and original local run.
Cost 0.1875 GPU h; native exit 0. Seven new diagnostic rows were finite:
max continuity 2.06e-16, Gauss 3.00e-8, energy residual 4.63e-11 J.

Standard video/radio products: `results/e120_gh200_preview_20260920/`;
remote `runs/scratch_e120_gh200/products`. Video: 901 frames through 107.1 us,
37.542 s, 5868x2880, fully decoded and visually checked. Radio fields were finite
with increasing time; optical fluence (2.22e-16 relative error), arrival-time
and scaling checks passed. Short-record VLF/optical-delay qualifications remain.
ARM product setup is in `docs/OLIVIA_GPU.md`. Preview/setup/products cost
1.371 GPU h including retries; commands/accounting in `tmp/olivia_gpu_rounds_20260919/`.

## 2026-09-19 -- GH200 optimization rounds (Round 7 complete)

User rule: stop after each round, report **kept/rejected**, measured gain with
uncertainty and physics checks; update the performance section/table in the
architecture TeX guide; then ask before starting the next. **Round 7 complete; no further optimization authorized.** Keep the Astra Pro conversation open and
reuse it: https://chatgpt.com/c/6aae8c33-f144-83ed-9b52-e055fe50b764

Budget: at most 100 additional GPU hours, final 10 reserved for confirmation.
GPU jobs use accel. Short 16/32/64-core CPU references were separately authorized.
Original local capture remains stopped at complete chk042480. Evidence and
runnable analysis: `tmp/olivia_gpu_rounds_20260919/`, mirrored under remote
`warpx_rrea_smoke/runs/gpu_rounds_20260919/`. Baseline binaries remain in
`gpu_cuda_mpi/`; retained candidate is in `gpu_route_candidate/`.

**Round 0 -- completed profiling; no new code.** Pooled/heap/forced-spill job
2291758, CUDA trace 2291764, host sampling 2291770. Existing 13 KiB/history
arena had no mature-workload spills and was about four times faster than
heap-only in the short probe. A forced 1 KiB pool exercised 14637125 spill
allocations; bulk probes agreed within 1e-10 relative, continuity near 1e-16.
Arena allocation requests are not CUDA allocation API counts. The old phase
header was unreferenced; the later audit removed it and its unused patcher.
Nsight collected about 5.9 s of kernels during 50.8 s of evolution; overlapping
durations are not a wall-time partition. User CPU samples placed 31394/48309
(65.0%) in AddToRecipientCell's local-box scan, with symbolized loop/contains
IPs. Kernel-mode samples were unavailable, some backtraces broken; CUDA 12.6
profiler libraries warn about driver 13.0. Timings corroborate the intervention.

**Round 1 -- KEPT configuration; no source addition.** Screen 2291761 compared
GPU kinetics and host kinetics with 1/12 transport threads. Six counterbalanced
pairs (2291768) retained host12 with GPU fluid/Maxwell: 10.1982 -> 9.1720 s/step,
1.11188x speedup, paired 95% CI [1.11000,1.11376] (10.1% less time).
Eight continuation seeds 7100101--7100108 (2291769) versus local six-rank CPU
passed all 76 checks; largest sampled feedback interval endpoint 2.78%.

**Round 2 -- KEPT: last-valid-box shortcut.** Nine production lines, no flag.
Try the previous valid box before the original scan. Disjoint immutable valid
boxes preserve recipient choice; grown-box precedence, off-rank routing,
arithmetic, RNG and validation are unchanged. Astra Pro approved this patch
unchanged. One focused existing-smoke case covers warm reuse/reset, CIC box
boundaries, order-sensitive sums and an empty-rank emitter. It passed original
and cached CPU code on 1/2/4 ranks; local full production build and all 28
native tests passed. Separate CUDA build 2291806 passed all 25 tests; updated
GPU accumulator smoke passed before confirmation.

Six counterbalanced pairs (2291858), holding host12 fixed: **8.43908 ->
2.05090 s/step; 4.11483x, 95% CI [4.10675,4.12292], 75.7% less time**.
Each run has 20 steps; discard five; paired runs, not individual steps, supply
the uncertainty unit. Control times differ across allocations, so do not
multiply R1/R2 ratios or interpret this interval as across-node uncertainty.
Eight matched seeds (2291859) passed all 76 physical-equivalence checks:
1% bulk and 5% sampled energy-band/feedback bounds with intervals inside the
bounds. Largest R2 endpoints: 0.285% bulk, 2.84% feedback; continuity <=2.09e-16,
Gauss <=3.01e-8. Reference pair-production and annihilation increments exceed
200 events per continuation. This does not establish every rare tail or true
model validity. `r2_paired_summary.json` and `r2_cache_physics_comparison.json`
contain the estimates. Cache plus GPU kinetics took 3.041 and 3.010 s/step in
two probes, so host12 remains faster. A 100-step replay (2291909) completed at 2.075 s/step after five warm-up
steps; finite diagnostics, continuity <=1.34e-16 and Gauss <=3.00e-8. That
controller then failed before fresh evolution because transport threads need
the typed driver option. Retry 2291921 stopped on its nonempty preparation
directory; corrected launch 2291924 completed the fresh 40-step capture.
Use `--transport-omp-threads 12`, not a protected extra key. The fresh
chk000040 was downloaded and continued successfully through step 41 on the
local CPU; evidence is `r2_cpu_return/`. Both final captures had finite reduced
diagnostics. No simulation or build is left running locally.

**CPU references -- completed, pre-cache code.** Build 2291759 passed 28
checks; 2291763 used one AMD EPYC 9745 node, one thread/rank, order 16/32/32/16.
16 cores: 3.545, 3.437 -> 3.491 s/step; 32: 1.918, 1.815 -> 1.866 s/step.
Build and replays cost 4.29 billed CPU hours (including memory billing).
The architecture TeX table compares these with local CPU and 1/2/3 GPUs on
the same baseline code and timing window. Added 64-core job 2291948 used the same pre-cache executable on another
EPYC 9745 node (c1-3): 2.19009, 2.17797 -> 2.18403 s/step, 17.0% slower
than the earlier 32-core mean. No continued scaling is demonstrated. Both
replays completed with finite diagnostics, continuity <=1.41e-16 and Gauss
<=3.00e-8; cost 1.8 billed CPU hours. No new CPU build was needed.
CPU and multi-GPU timings have not been refreshed with the cache; do not silently combine code versions.

At the end of Round 2, no Olivia jobs remained. Rounds 0--2 used about
**2.14 GPU hours**
including builds and failed launch preparation; project total is 3.47 used,
996.53 available. CPU billing is 258957.04 used, 1042.96 available. Recheck
quota before future submissions.

**Round 3 -- REJECTED: phase-local count/offset/summary buffer reuse.**
The supplied six-file patch (+45/-11 including its repeated-call smoke) kept
state, arenas, device logs, variable outputs, kernels and validation unchanged.
Local full production build passed all 28 checks. CUDA job 2291935 passed
25/25 native checks plus explicit heap-only, 13 KiB pool and forced 1 KiB spill
checks (73987 spill allocations). Production timings used GPU kinetics, where
the candidate is active, and held the accepted R2 routing cache fixed.

Six adjacent counterbalanced pairs, 20 steps each, excluding five warm-up:
**3.00677 -> 3.04149 s/step; speedup 0.98876, paired 95% CI
[0.96737,1.01063]**. The point estimate is about 1.1% slower; no significant
gain or slowdown is established, and the interval does not reach the required
1.02x gain. All six-file candidate changes, including extra smoke logic, are
removed; the R2 routing cache is retained. All twelve production replays
had finite reduced diagnostics, continuity <=1.86e-16 and Gauss <=3.00e-8. The larger eight-seed campaign was not run
because performance failed. This rejects this optimization, not every possible
allocation change; the measured result does not establish allocator dominance.
Exact patch, commands, paired timings and analysis live in the evidence directory
(`round3_metadata.diff`, `round3_confirm.py`, `r3_paired_summary.json`).
Restoration completed: local full build passed 28/28 native checks; Olivia
rebuild 2292043 passed 25/25. The accepted executable and prepared tree are
back to R2; no metadata-reuse code remains. No local simulations/builds or
Olivia jobs remain. R3 build/timing/restoration consumed 0.434 GPU hours; the
64-core reference used 1.778 CPU billing hours. Current project accounting:
3.90 GPU hours used, 996.10 available; 258958.82 CPU hours used, 1041.18
available (17:17 CEST). Rounds 0--3 total about 2.57 additional GPU hours.

Both performance tables were reformatted and compiled/visually checked.
**Round 4 -- KEPT configuration: 48 host transport threads on one GH200.**
No executable code change. The existing R2 binary (28 local/25 CUDA native
checks passed) uses CPU kinetics, GPU fluid/Maxwell, global OMP_NUM_THREADS=1,
and OMP_DYNAMIC=false. Only the scoped transport-thread setting varies.
Screen 2292061 used a fixed 48-core allocation: 6/12/24/48 threads averaged
2.43151/2.05536/1.89367/1.81100 s/step, two reversed-order repeats each.
Initial controller 2292060 failed before evolution; its path expression was
fixed and command loops checked locally. Upper screen 2292064 used a fixed
72-core allocation: 48/72 threads averaged 1.78024/1.76054 s/step. Only 1.1%
extra speed at 72; select 48 for confirmation without changing affinity policy.

Six fresh counterbalanced pairs (2292092), fixed N=6 before viewing results,
20 steps/run, discard five: **2.04908 -> 1.80203 s/step; speedup 1.13731,
paired 95% CI [1.11144,1.16377]**. This is 13.7% more throughput or 12.1%
less time. Screen data are excluded; the interval is not across-node uncertainty.
The unscored runtime audit showed exactly 12/48 level-1 workers, both allowed
on CPU mask 72-119; no OMP_PLACES/GOMP_CPU_AFFINITY, binding false, dynamic
false, unrestricted thread limit. Allowed masks do not prove residency.

Eight matched seeds (2292093) passed all 76 physical comparisons: 1% bulk,
5% sampled bands/feedback, with 95% intervals inside limits. Largest endpoints:
0.313% bulk, 2.23% bands/feedback; continuity <=2.19e-16 and Gauss <=3.01e-8.
A 100-step replay on another node (2292113) averaged 1.79197 s/step after five;
finite diagnostics, continuity <=1.84e-16. Fresh 40-step capture completed;
its downloaded checkpoint continued on local CPU through step 41.

Astra Pro reviewed current source and R2/R3 evidence in review_round4.zip.
It found no thread-count dependence in history assignment, RNG keys, newborn
lineage or ordered replay. Its requested runtime audit is above; the existing
analysis already scores steps[5:], not the all-step mean field. N=6 was frozen
rather than extended based on the observed CI. Source review is not a runtime
test or physical-model validation. Corrected one stale direct-replay comment
in RreaTransportLog.H: all thread counts actually collect logs first.

Accepted launch: `CPUS=48`, `--transport-omp-threads 48`, CPU kinetics with
GPU fluid/Maxwell. The full command is in docs/OLIVIA_GPU.md. No simulations,
builds or queued jobs remain. R4 consumed 0.459 GPU hours and no CPU billing
hours; rounds 0--4 total about 3.03 additional GPU hours. Project accounting
at 17:49 CEST: 4.36 GPU hours used, 995.64 available; CPU billing remains
258958.82 used, 1041.18 available. Recheck quota before another submission.
The updated, visually checked PDF is in evidence report_round4/ and at the
canonical docs path. TeX and operating instructions are current.
**Round 5 -- REJECTED: zero OpenMP worker spinning. No source change.**
Fresh profiling of the accepted one-GH200/host48 configuration (2292574)
collected 25408 main-thread samples: 27.2% in AddToRecipientCell, 6.8% in
AddToField; 14.0% unresolved. Workers had 102121 samples, 32.2% in two
libgomp barrier functions. These are on-CPU samples, not wall-time fractions.
A separate delayed CUDA capture recorded 992219 kernels (5.084 s summed),
0.0248 s of copies and 4.051 s of CUDA API calls across about 23.6 s;
overlapping durations must not be added. Kernel-mode CPU sampling was
unavailable; Nsight 2024.4 warns that its CUDA 12.6 tracing libraries predate
the installed driver 13.0. Initial timed capture 2292475 lacked CPU samples
and stopped before 20 steps; excluded. Replacement captures ran to completion.

The single hypothesis was GOMP_SPINCOUNT=0 versus 300000, all other settings
fixed. Astra Pro supported the bounded test, not a bottleneck claim: reduced
spinning can lose to sleep/wakeup overhead. Its GCC-source review also explains
that displayed PASSIVE can mean an unset wait policy; no wait-policy override
was introduced. Actual executable audits verified the two spin counts and 48
workers in the same CPU mask 0-47, with dynamic teams and binding both false.

Six fixed-N counterbalanced pairs (2292597), 20 steps each, discard five:
**1.830113 -> 1.828283 s/step; 1.001007x, paired 95% CI [0.988001,1.014184]**.
The 0.1% estimated throughput gain fails the 2% retention threshold; no useful
gain is established. Keep the R4 configuration, without a spin-count override.
All twelve replays completed with finite reduced diagnostics, continuity
<=1.85e-16 and Gauss <=3.00e-8. The extra eight-seed campaign was skipped after
performance failed; these diagnostics are not a new full equivalence claim.
No production code, test or experimental launcher setting was retained.
Evidence: round5_confirm.py, r5_paired_summary.json, r5_production_diagnostics.json
and r5_profile_summary.json in the campaign directory. The fresh profile leaves
serial deposition and many small fluid kernels as leads, not proven fixes.

All Olivia jobs completed; R5 used 0.209 GPU hours, no CPU billing hours.
Rounds 0--5 used about 3.24 additional GPU hours. Accounting at 18:50 CEST:
4.57 GPU hours used, 995.43 available; CPU billing unchanged at 258958.82 used,
1041.18 available. Performance TeX/table and canonical PDF updated, compiled
and visually checked. No local simulation/build remains. Astra Pro tab stays
open.

**Round 6 -- KEPT: two-entry valid-box cache.**
Table 1 now includes the retained R6 result (1.705 s/step, 4.17x versus the
historical local CPU); older configurations are explicitly historical.
CLAUDE.md requires both performance tables and the PDF to stay current.
Eight net production lines extend the R2 hint to the two most recently used
valid boxes, swapping on a second-entry hit. This avoids scans when CIC
recipients alternate across a boundary; grown-box precedence, deposition
arithmetic and serial replay order are unchanged. No new runtime flag.
Astra Pro reviewed the exact source/diff in review_round6.zip and requested
no correction. The existing smoke gained three net lines and four slabs to
exercise alternating hits, eviction of both hints, reset, CIC boundaries,
order-sensitive sums and an empty-rank emitter. Baseline and candidate passed
that smoke on 1/2/4 local ranks. Both full local production builds passed
28 native checks; the candidate CUDA build passed all 25 checks.

Six fixed-N counterbalanced pairs (2292621), 20 steps/run, discard five,
host48 with GPU fluid/Maxwell and CPUS=48: **1.868109 -> 1.705078 s/step;
1.095368x, paired 95% CI [1.063481,1.128211]**. This is 9.5% more throughput
or 8.7% less step time. Both arms used the same GPU and CPU mask 72-119;
no R5 spin override. All twelve reduced diagnostics were finite.

Eight matched local-CPU seeds (2292670) passed all 76 physical comparisons:
1% bulk and 5% sampled bands/feedback bounds, with intervals inside them.
Largest 95% interval endpoints: 0.303% bulk and 2.58% bands/feedback;
continuity <=2.14e-16, Gauss <=3.01e-8. These are implementation-equivalence
checks for the sampled observables, not physical-model validation.
The 100-step replay completed through 42580 at 1.63870 s/step after five
warm-up steps; scheduled diagnostic rows at 42500/42550 were finite,
continuity <=1.88e-16 and Gauss <=3.00e-8. The longer replay is not another
paired speedup estimate. Earlier checkpoint portability checks remain relevant:
this round changes only advisory routing indices, not checkpoint data.

Evidence: round6_cache2.diff, round6_confirm.py, r6_paired_summary.json,
r6_cache2_physics_comparison.json and r6_long_summary.json in the campaign
folder. Retained executable remains in gpu_route_candidate/build; launch
settings are unchanged. R6 cost 0.389 GPU hours, no CPU billing hours;
rounds 0--6 total about 3.63 additional GPU hours. Accounting at 19:42 CEST:
4.96 GPU hours used, 995.04 available; CPU billing unchanged at 258958.82 used,
1041.18 available. No queued/running Olivia jobs. Journal and performance
TeX/table and canonical PDF updated, compiled and visually checked. No local
simulation/build remains. Stop before Round 7; Astra Pro tab remains open.

**Round 7 -- completed scaling; keep one GH200 as the default.**
No code change; commit 7af592c, retained R6 cache, CPU kinetics with 48
transport threads per rank and GPU fluid/Maxwell. Six fixed-N counterbalanced
pairs (2292781), 20 steps/drop five on the same two-GPU allocation:
**1.671408 -> 1.556137 s/step; 1.074092x, paired 95% CI
[1.065602,1.082649]**. Efficiency 53.7%; production cost 1.671 -> 3.112
GPU-s/step (+86.2%). Extra GPU cost is not justified as the default for 7.4%
more throughput. Whole-invocation means including startup were 38.20/37.73 s.
Both Table 1 current rows now use these fresh R7 measurements.

Actual audits verified 48 workers/rank, global OMP1, dynamic false. Every timed
one-rank arm used PCI 0009:01:00.0/CPUs 0--47; rank 1 of the two-rank arm used
PCI 0019:01:00.0/CPUs 72--119. Topology confirms locality, not the MPI transfer
path. Astra Pro approved the setup: this scales the whole hybrid configuration
(48 to 96 CPUs), and WarpX prints its I/O-rank interval, not an MPI maximum.
Memory residency and node exclusivity were not controlled.

All 76 eight-seed comparisons (2292877) passed: largest 95% interval endpoints
0.303% bulk and 2.65% bands/feedback; continuity<=2.40e-16, Gauss<=3.01e-8.
Evidence: round7_scaling.py, r7_paired_summary.json, r7_scaling_diagnostics.json,
r7_g2_physics_comparison.json. Both tables and PDF updated. R7 cost 0.435 GPUh;
rounds 0--7 total about 4.07 GPUh. Account 21:03 CEST: 5.39 GPUh used, 994.61 available;
CPU billing unchanged. Scaling jobs finished; no Round 8 started.

## 2026-09-19 -- Olivia GPU deployment and scaling

Branch `gpu-cuda`, numerical source `c77e83b`; only a misleading CPU-only
startup banner was removed afterward. The original local E120 capture remains
stopped, with complete `chk042480`. The clean local production build passed
28 native checks; Olivia MPI+CUDA build `2291532` passed 25. The ordinary
two-GPU capture launcher passed (`2291711`), and placement probe `2291617`
verified four distinct GPUs. Launcher/transfer checks: 12 passed under WSL.

**Scaling results and interpretation live in the architecture TeX guide,
section ?Execution backend and measured GPU scaling?.** Two repeats each:
1 GPU job `2291719` on gpu-1-6; 2 GPUs `2291720` on gpu-1-76; 3 GPUs `2291721`
on gpu-1-11. Four-GPU job `2291702` was cancelled while pending at the user's
request (32-minute estimated wait), consuming no allocation time. All measured
runs completed 20 steps with empty stderr and distinct GPUs per rank.

Raw timings, commands, telemetry, conservation diagnostics, accounting and
`summarize.py` are in `tmp/olivia_gpu_scaling_20260919/`; remote counterparts
are under `warpx_rrea_smoke/runs/gpu_scaling_20260919/`. The shared checkpoint
and seed schedule are in remote `runs/gpu_bench_20260919/`. These are detached
replays, not resumable captures. Original local source/capture paths are
`/home/david/rrea_olivia_gpu_check_20260919` and
`/home/david/rrea_runs/e120_local_head/capture`.

The earlier non-MPI GH200 experiment (`2291286`) averaged 10.056 s/step versus
6.968 on local CPU over steps 42486-42510. This is a different timing window
and executable from the scaling table; its evidence is in
`tmp/olivia_gpu_20260919/`. Transport occupied 93% of the first GPU profile;
this identifies the expensive path, not a particular kernel or host phase.
The short driver capture (`2291407`) produced `chk000040`, downloaded through
`pull-checkpoint` and successfully continued through step 41 on local CPU.
The operational guide records the seed-rate and diagnostic-cadence settings
needed for short captures.

At campaign close all jobs are finished or cancelled. Every Olivia job used
`accel`; no CPU benchmark ran there. **GPU hours: 1.33 used, 998.67 available;
CPU billing unchanged at 258952.75 hours.** Live quota must be rechecked before
future submissions. Build/run/configuration/download instructions are in
`docs/OLIVIA_GPU.md`; default executable is in `gpu_cuda_mpi/build/bin/`.

## 2026-09-17 -- GPU performance campaign, rounds 1-8

Adversarial review loop against GPT 6 Astra Pro (`docs/CHATGPT_REVIEW_LOOP.md`),
branch `gpu-cuda`, RTX 4060 Ti / WSL2 / CUDA 12.6. Detail and current state in
`docs/GPU_PORT.md`; this is the narrative.

**Where it started.** Every benchmark in the first two dozen rounds ran with
`gpu_fluid=0 gpu_maxwell=0 kinetic_backend=cpu` -- a configuration that does
not ship. The real shipping deficit was 6.7x against the single-core CPU
build, not the 2.8x being quoted. The equivalence gate pinned the same
switches off, so the recorded "physics equivalence PASSED" covered a CUDA arm
nobody runs. Both are fixed; the gate now defaults to shipping values and
reads all three switches back out of `warpx_job_info`.

**The two things that worked.**

1. `rrea.kinetic_backend = cpu` in the production deck. GPU kinetics costs
   +12.105 s of a 22.4 s run at 256x128 and has never won at any measured
   scale. The shipping configuration went 24.70 s -> 11.92 s, **2.07x**, and
   the deficit against the single-core CPU build fell from 6.7x to 3.2x.
2. A per-history device arena. The advance kernel's cost was device-heap
   allocation: 7,679,620 in-kernel `new`/`delete` pairs across 221,257
   histories in a ten-step run, ~35 per history. Giving each history a private
   13 KiB bump-pointer slice took GPU kinetics from 25.400 +/- 0.276 s to
   16.742 +/- 0.253 s, **1.517x**. It still loses to host kinetics (1.107 at
   the production mesh), so it changes no default.

**What was ruled out, with measurements, so nobody retries it.** Kinetic batch
size is not the lever and larger batches are sharply worse (19.88 s at 32768
against 10.76-11.27 s at the 8192 default). Launch overhead, host
repatriation and the population branch are all dead: the phase timers put
`advance_wait` at +8.752 s of a +8.763 s excess while `advance_submit` is
0.127 s, `unpack_consume` 0.087 s and `map_live` + `weight_commit` under
0.01 s together. `ComputeSpaceChargeField` is not a wasted solve --
`BeforeFieldSolve` returns true and the ordinary solver is skipped. The
four-field transport-scratch device-arena flag is a null (1.005, CI
0.932-1.078); the larger workspace flag has never been timed and that null
must not be quoted for it.

**Method failures worth remembering.** A particle-scaling test measured
nothing for a full round: it varied `expected_seed_macro_particles`, a
constant the deck defines and never reads, and reading the value back out of
`warpx_job_info` proved only that it was *passed*. Assert on an observable.
`tiny_profiler.print_threshold` defaults to 1.0 and folds cheap regions into
`Other`, so an absent region establishes nothing -- pass 0 before reading a
call count. And a gate whose command line carried literal `\n` sequences
passed two stray arguments for a whole acceptance round.

**Open at the pause.** The spillover patch (round 8) is applied and compiles
but is **not yet gated**: heap-vs-pool ownership tracking, `arena_kib=13` and
`arena_spill=1` as defaults when GPU kinetics is selected. Its acceptance
cases are listed in `docs/GPU_PORT.md`, and the forced-spill gate at K=1 is
the only one that exercises the fallback branch. After that, in order: a fresh
phase profile of pooled kinetics, the fused reaction-conservation audit, the
batched diagnostic sums, then a tables-only device mirror if advance still
dominates. The pre-arena plumbing overhead is explicitly unmeasured.
