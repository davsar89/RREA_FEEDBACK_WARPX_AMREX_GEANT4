# External review and triage

**Reviewer.** GPT-6 Pro on chatgpt.com, 2026-11-09. It was given the repository at commit
2973eb0, the page and its tools, every cited WarpX-26.06 and AMReX-26.06 source file, and
review instructions (correctness against the source, ambiguity, giveaways, references,
pedagogy, bugs and accessibility). It could read the code but not build it, so it did not run
the native probe or the browser suite.

**Result.** 59 findings: 27 high, 29 medium, 3 low. Each one was checked against the cited
source before anything changed. 54 were applied as proposed or in an equivalent form, and 5
were applied in a reduced form, with the reason given below. None was rejected: every finding
pointed at a real defect, although a few overstated it.

**Reviewer's overall impression,** quoted:

> The course gives a newcomer concrete code to follow from the particle push through charge
> deposition and the field solve. The most consequential corrections concern which stage owns
> an update, which quantities are conserved, and when a diagnostic measures avalanche growth.
> The content validator passed all 140 questions across 14 lessons, and all ten examples
> runnable without the native drag probe reproduced their embedded output. I checked page
> behavior from the JavaScript and CSS. The native probe and full headless browser suite were
> unavailable here.

## Findings

The finding is the reviewer's, condensed. Paths are shortened: `O/` is
`native_amrex_rrea/warpx_overlay/Source/Rrea/`, `N/` is `native_amrex_rrea/include/rrea/`,
`W/` is `upstream/WarpX-26.06/Source/`, `A/` is `upstream/AMReX-26.06/Src/`, `guide` is
`docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex`.

| # | Location | Sev. | Finding | Verdict | Change |
|---|---|---|---|---|---|
| 01 | pop3 | high | Options 0 and 2 were both true: thinning keeps the four CIC charges and each survivor's own kinetic energy. | applied | Option 2 is now the group's total weighted kinetic energy, which thinning keeps only on average. |
| 02 | val equation 1, intro, cd caption | high | The integral subtracted below-threshold segments; `rrea_profiled_atmosphere.py:524-549` integrates only the positive excess. | applied | All three now use ∫max(\|E\| − Tn, 0)/7300 kV dz. |
| 03 | big runaway lab, tab density lab | high | At 16 km the labs used the 15 km density, because the profiles were sampled on a coarse grid. | applied | `build.py` samples field and density at their own knots plus the domain ends, which is exact for these piecewise-linear profiles. |
| 04 | big1 key and explanation | high | Field against drag cannot say that a given electron gains energy: one moving across the field gains nothing. | applied | Key: "eE exceeds the ensemble drag at its energy, so it can keep gaining energy"; the explanation says the gates classify an electron as runaway capable whatever its direction. |
| 05 | kin5, kin intro, equation 3 | high | Rates are not frozen at the midpoint energy for a whole substep with several events. | applied | The midpoint freeze applies to the first event search only; later searches refreeze at the survivor's energy (`RreaChargedInterleave.H:489-494`). |
| 06 | fld3 key and why[1] | high | "Before committing any state" promises a rollback that the evolution loop does not provide. | scaled | The engine does abort before the coefficients are used, as its own comment says, so the claim was imprecise rather than false. Key: "The run aborts collectively before the out-of-range coefficients are used". |
| 07 | amr3-5 stem and key | high | Full `Redistribute` is used for None, HybridPIC and refined levels, not only for None. | applied | The stem is restricted to a single level; the explanation names HybridPIC and refined levels; reference and excerpt now run to `WarpXEvolve.cpp:749`. |
| 08 | wx1 intro and step diagram | high | The OneStep box said "gather, push, deposit"; in ES mode the push skips deposition and the hook deposits in BeforeFieldSolve. | applied | Diagram rows "OneStep: gather and push (no deposit)" and "BeforeFieldSolve: deposit, solve"; intro and caption explain `skip_deposition`. |
| 09 | amr1-10, val9 | high | A rank-count change also folds rank state; the restart recipe omitted the mandatory reason flag. | applied | val9 key names `--restart-epoch-break` plus `--restart-epoch-reason`; amr1-10 and the val restart text mention the fold; policy references fixed. |
| 10 | pop9 key and explanation | high | A strict-ceiling abort alone does not prove fivefold electron growth: any species can trip it, at a projected count, with equality depending on mode. | scaled | The project documents this abort as a result in C&D mode (`rrea_cd_physics.py` docstring, guide 1794-1796), so the reading stays; the key and explanation now state the projected count, any species, equality only in C&D plane-flux mode, and that multiplication is measured by the plane fluxes. |
| 11 | wx2-10 key | high | The key listed a missing ambient field as a limit of lab-frame electrostatics, but WarpX adds external fields after the solve. | applied | Key: "It acts instantly: no finite propagation speed, no B field and so no radiation". |
| 12 | fld10 explanation | high | At the 16 km top the 37 kV/m fringe is close to the avalanche threshold and above the drag minimum. | applied | The explanation gives 91% of 276n there (276n ≈ 41 kV/m) and the 32 kV/m drag minimum; three new fact checks in `build.py` bind these numbers. |
| 13 | kin intro, takeaway 2, kin2 | high | Only the optical-depth driver aborts above 8192 substeps; the GS and impulse drivers clamp. | applied | Intro, takeaway and kin2 why[3] say so (`RreaKineticStep.H:664-693`). |
| 14 | kin chord caption, takeaway 3 | high | "Energy changes only by line-integral work" dropped drag and discrete transfers. | applied | Caption and takeaway list field work, drag and discrete events. |
| 15 | big1 why[2], hook5 | high | Real escapes were attributed to WarpX's boundary removal; the transport books them and invalidates the parent. | applied | Both say the transport records the escape at the exact crossing and retires the parent, so WarpX never sees the overshoot; hook5 cites `RreaKineticStep.H:979-995, 1248-1251`. |
| 16 | tab8, tab takeaway | high | "Photons that annihilate below the cut": positrons annihilate, and annihilation photons carry at least m_e c²/2 ≈ 255.5 keV. | applied | Explanation and takeaway corrected; the guide sentence it came from is listed below. |
| 17 | wx1-4 why[2], wx1 takeaway | high | WarpX does have a plasma-frequency limit, `max_omegap_dt`, in its adaptive update. | applied | why[2] and the takeaway name it and when it applies; references to `WarpXComputeDt.cpp`. |
| 18 | wx1-5 explanation | high | Native ρ was grouped with the retained E pointers and copy-back targets. | applied | The explanation separates what the coupling reads from what it writes back. |
| 19 | wx2 menu, resampling row | high | The row put RREA thinning after the secondary commit. | applied | "thinning inside the secondary transaction, refill splits after it; cell charges stay exact". |
| 20 | amr1 intro, box figure, example | high | "92 boxes per rank" is a mean; the distribution map balances cells, so boxes per rank differ. | scaled | Labelled as an average everywhere, and the caption says the space-filling curve balances cells, not boxes. Porting AMReX's mapping to print the true per-rank counts would add code for no teaching gain. |
| 21 | pic intro, takeaway, pic1 | high | The two magnetic half-steps were presented as WarpX's general update; that is the finite-difference path. | applied | Text names the finite-difference solver and contrasts PSATD. |
| 22 | amr3-8 why[2], amr3-3 | high | Checkpoints keep runtime components only when their communication flag is on. | applied | Both state the flag, and cite `FlushFormatCheckpoint.cpp` and `ParticleIO.cpp`. |
| 23 | val5 key and takeaways | high | The ≥ 1 MeV flux cancels only equal-state recrossings. | applied | Key: "A selected electron that crosses a plane and comes back unchanged cancels"; the intro says "equal-state recrossings". |
| 24 | amr2 CIC lab | high | Ruler and particle were drawn half a cell left of the cells used in the calculation. | applied | One transform for cells, ruler, edge and particle. |
| 25 | fld relaxation lab, pic Courant lab | high | Markers were clipped at the axis edge while labelled with their true values. | applied | The relaxation axis spans every slider value; the Courant axis grows with the Courant number. |
| 26 | kin8 why[1] | high | "No finite set of samples has a mean of exactly ½" is false. | applied | Reworded to what excluding the endpoints does and does not do. |
| 27 | wx1-8 why[2] | high | "Seeds appear every step" confused the schedule check with injection. | applied | The hook checks the schedule every step and injects only the events due. |
| 28 | Mixed practice, review | medium | Starting either mode deleted saved answers before any attempt. | applied | Practice attempts are kept with the session's frozen list; saved answers change only when answered. The browser test checks this. |
| 29 | Skip link | medium | The skip link routed to the overview. | applied | It focuses the content without routing; new browser test. |
| 30 | val7 key and hint | medium | The 5% gap is in log growth; in M the shortfall is larger. | scaled | The question's point (the cut is converged and is not the cause) stands, so only the metric is named: log growth 0.9465 and 0.9627 of the references, M = 23.1. |
| 31 | pic4 | medium | Four is the stencil size; one, two or four points receive nonzero charge. | applied | Stem asks for "at most"; the explanation covers a particle on a grid line. |
| 32 | pic10 | medium | Halving the count and doubling the weight does not by itself fix the noise change. | applied | Stem now samples the same distribution with independent macroparticles. |
| 33 | kin equation 2, intro | medium | The channel share is conditional on an event, and the code draws −ln(1 − u). | applied | Equation shows −ln(1 − u) and P(c \| event). |
| 34 | val lesson | medium | The analytic interval and the measured flux planes differ: the planes are inset. | applied | The intro explains the inset planes. |
| 35 | pop6 | medium | Equality at the cap depends on whether the C&D plane diagnostic is on. | applied | Explanation states both policies and the equality rule. |
| 36 | pop4, pop reflection | medium | Unbiased weights do not make the target the only variance control. | applied | Text separates bias from variance. |
| 37 | wx2-6 | medium | Axisymmetry alone does not force Jθ = 0. | applied | Key: "The model is axisymmetric with Jθ = 0, so only Er, Ez and Bθ are kept". |
| 38 | hook9, hook takeaway | medium | The transport-completed check was presented as a precondition for every intercepted solve; the initial solve is exempt. | applied | Stem restricted to evolution; explanation covers the initial solve. |
| 39 | amr2 takeaway, amr2-1 | medium | CPU tiling is opt-in; a default MFIter visits whole boxes. | applied | Both name `TilingIfNotGPU()`. |
| 40 | amr1 intro, takeaway, glossary | medium | A BoxArray need not tile the whole domain (refined levels). | applied | Definition fixed in intro, takeaway, reflection and glossary. |
| 41 | fld intro | medium | The 10⁻⁶–60 Td closure is not all measured; the low-field end is a model. | applied | The intro gives the closure's provenance. |
| 42 | fld5 | medium | A passing continuity gate is a tolerance test, not exact continuity, and does not bound Gauss residual growth. | applied | Explanation states the tolerance and that Gauss is monitored. |
| 43 | big6 why[1], hint | medium | A local demotion is a judgement under present conditions, not proof the electron can never run away. | applied | Reworded. |
| 44 | obs8 | medium | The stem asked for counts; the citation pointed at video policy. | applied | Stem asks for a count rate; reference `AGENTS.md:57-60`. |
| 45 | obs9, obs10, val9, val10 | medium | Policy references missed the operative rules. | applied | References corrected. |
| 46 | tab5, tab4 | medium | The figure showed A/B/C while the question named channels; references did not show the crossover. | applied | Channel names shown, crossings still hidden; tab5 cites the crossing computation in `build.py`, tab4 the runnable example. |
| 47 | wx2-4 | medium | The deck selects Higuera–Cary but does not prove its properties. | applied | Explanation cites Higuera and Cary (2017) and the selection enum. |
| 48 | amr2-4 | medium | The cited SumBoundary overload was not the one the coupling calls. | applied | Reference is the overload with source and destination ghost extents. |
| 49 | pic3, pic8, amr1-2, kin2 | medium | References did not substantiate the claims. | applied | References replaced. |
| 50 | hook intro, hook1, hook10 | medium | The native library also needs zlib, and CMake edits use CMake conditions, not `#ifdef`. | applied | Text and hook1 key corrected. |
| 51 | Plot keyboard readout | medium | Arrow-key values were not announced. | applied | A polite live region announces each point. |
| 52 | Navigation drawer | medium | Widening past the breakpoint left the content inert. | applied | Leaving the narrow layout closes the drawer; new browser test. |
| 53 | big3 | medium | The explanation attributed the restricted-stopping reading of "1.7 MeV" to the guide. | scaled | The two minima (1.65 MeV restricted stopping, about 1.15 MeV ensemble drag) are correct, so only the attribution changed: the match to restricted stopping is now flagged as an interpretation. |
| 54 | amr3-9 why[1] | medium | Rigid injection is chosen by `particles.rigid_injected_species`, not `species_type`. | applied | Corrected, with `MultiParticleContainer.cpp` references. |
| 55 | amr3-10 | medium | Only the handshake is collective; payloads move point to point. | applied | Corrected, with `AMReX_ParticleCommunication.H` references. |
| 56 | val profile caption | medium | The shaded interval omits the first 50 m above the threshold crossing. | applied | Caption reworded. |
| 57 | amr2-3 hint, val5 and pic6 distractors | low | Hint named the answer; two distractors were easy to eliminate. | applied | Hint and both distractors rewritten. |
| 58 | Option order | low | Every learner saw the same option order. | applied | A per-learner seed, stored locally, shuffles the options; stable across reloads. |
| 59 | wx2-9 why[3] | low | "Checked on every build" was unclear: the check runs at startup. | applied | Reworded. |

## Left for the project's decision

The review led to three statements in the architecture guide that the course no longer
repeats. They are outside the course and were not changed:

- **Guide lines 828-830:** "Photons that Compton-degrade or annihilate below the same cut".
  Positrons annihilate, not photons, and each annihilation photon carries at least
  m_e c²/2 ≈ 255.5 keV, far above the 50 keV cut.
- **Guide lines 1773-1774 and 2205:** the C&D integrand is written as E − Tn without
  max(…, 0). `estimate_integrated_avalanche_lengths` integrates only the positive excess,
  which matters for T = 284 near the ends of the 276 interval.
- **Guide line 859:** "the ≈1.7 MeV friction minimum that sets the RREA threshold". The
  minimum near 1.65 MeV is in the restricted collision stopping; the ensemble drag that sets
  the force balance is minimal near 1.15 MeV. The Sternheimer argument there holds at both.
