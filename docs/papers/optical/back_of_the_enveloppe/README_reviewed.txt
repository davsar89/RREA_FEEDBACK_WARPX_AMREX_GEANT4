337 NM RREA CALCULATION: REVIEWED SHORT NOTE
22 September 2026

FILES
  rrea_337_back_of_envelope_reviewed.tex   Self-contained LaTeX source
  rrea_337_back_of_envelope_reviewed.pdf   Render-checked compiled note (6 pages)
  rrea_337_review_check.py                Python 3 standard-library check/calculator
  rrea_337_review_check_results.json      Output from the default invocation
  README_reviewed.txt                     This file, including source-code audit

BUILD
  pdflatex rrea_337_back_of_envelope_reviewed.tex
  pdflatex rrea_337_back_of_envelope_reviewed.tex
No external figures, bibliography file or BibTeX are needed. Standard TeX Live
packages include newtx, tcolorbox, hyperref and the other packages in the preamble.

CALCULATE
  python rrea_337_review_check.py
  python rrea_337_review_check.py --distance-m 8000
  python rrea_337_review_check.py --deposition-fraction 0.5

SCOPE OF REVISION
The original in-field arithmetic is retained. The note now distinguishes:
  (a) in-field fluorescence: 3.00e18 photons; 1.768 J;
      peak irradiance 1.122e-3 W/m2 at 1 km;
  (b) an added, conditional post-exit component;
  (c) their sum ONLY when the same Gaussian time envelope and distance apply.
The default post-exit case uses an 8.3 MeV conditional mean above 1 MeV, a
21.59 ph/MeV extrapolated AIRFLY dry-air yield, and full kinetic-energy deposition
in the selected downstream volume. It gives 1.79e18 extra photons and a total
1.793e-3 W/m2 peak at 1 km. Full containment is an assumption, not a transport
result. No downstream term should be added if the same energy deposits and
fluorescence are already included in a simulation's in-field/effective yield.

The original 24-page guide was NOT silently overwritten. This package revises
the standalone calculation note. It explicitly flags which elements of that
older discussion need qualification.

REVIEW CHECKS
1. The original rounded numerical values pass the supplied 0.3% arithmetic
   tolerance, including avalanche length, quenching rate, survival probabilities,
   density factor, photon count, energy, peak, lifetime and boxcar factor.
2. Xu's per-metre yield is explicitly interpreted as emitted, already-quenched
   light. The paragraph does not itself restate the convention. The laboratory
   comparison supports the scale but is not independent author confirmation.
3. AIRFLY alpha=-0.36 changes the review's 23.91 ph/MeV result to 21.59 ph/MeV
   at 194 hPa and 216.65 K. The measurement interval was 240--310 K; this is an
   extrapolation. A universal 15% temperature correction is not justified.
4. The 13.1 MeV energy-loss check uses the same avalanche fit and simplified
   runaway-sector accounting. It does not establish a complete, independent
   energy-deposition audit of all particles in an applied field.
5. Forward electron-distance and actual path length must be matched to the
   optical yield. The note does not infer a universal 276/218 path correction.
6. The original optical.zip does not contain rrea_337_peak_estimate.py. The later
   thunderstorm_optical_guide_updated_source.zip DOES contain it (4,290 bytes).
   The older archive and the later generated bundle are different deliverables.
7. The September optical review's 50--70 mW/m2 peak is the SCATTERED curve.
   Its roughly 6e-6 J/m2 fluence is the UNSCATTERED comparison. Compare like
   quantities. At 7.4 km the present original baseline has 2.569e-9 J/m2
   unscattered fluence, a ratio of 2,336. The older plot's source normalization
   has not been established as equal to this scenario, so the mismatch alone
   is not proof of a code error. The full-containment addition still leaves a
   discrepancy of about 1,462 in fluence.
8. The sixfold ratio of Xu's highest/lowest broad-window yields is per metre.
   At fixed mature output count the effective avalanche length also changes,
   so sixfold event amplification does not follow from that yield ratio alone.

AUDIT OF THE SUPPLIED get_337 ROUTINE
Inspected original location:
  optical.zip -> optical/Lehtinen_codes/fgf_module.py
  (tracked in this repository as docs/papers/optical/Lehtinen_codes/fgf_module.py)

The source rate was normalized by c * NR * volume, using the routine's own
constants. Evaluation of its functions and an independent standard-library
replication give:
  q=1.00: whole modeled C-state yield = 0.380312 photons/electron-m
          direct 16 eV x 10 eV-window births = 111.352/electron-m
  q=0.24: whole modeled C-state yield = 0.365913 photons/electron-m
          direct window births = 26.7245/electron-m

Its coefficient can be reconstructed without running its feedback solver:
  eta_C = range(16 eV) * [d_sigma_i(7.3 MeV,16 eV) * 10 eV * n_air]
          * sigma_C * n_N2 * A_C / (A_C + alpha_C*n_air).
The standalone check implements this scalar replication. It is NOT a repaired
low-energy model. No supplied source files were edited.

The function uses representative 7.3 MeV primaries and 16 eV secondaries. It does
not evolve a full degrading/re-accelerating secondary-electron distribution.
The supplied Lehtinen thesis, Sec. 3.1, printed p. 42, explicitly states that its
approximation neglects ionization by primaries below the Monte Carlo threshold.
Dwyer & Babich (2011), Secs. 2, 5 and 6, describe full secondary production and
intermediate-electron source terms. These references support investigating the
missing cascade, but not equating every ion pair to a C-state excitation. Many
secondaries are born below the approximately 11 eV excitation threshold.

The code's C-state yield still needs a fraction selecting 337 nm from its whole
2P system. That fraction is NOT the f337=0.25 used for the separate combined
300--430 nm window. Thus a precise universal factor-of-15 repair is not supported.
A full or independently calibrated low-energy kinetic source is needed.

LIMITS
The 1.8 MeV cm2/g stopping power is reproduced as the review's rough assumption.
No new numerical ESTAR table has been obtained. The arithmetic check is not a
transport simulation, absolute optical calibration, confidence interval,
cloud-propagation calculation or FEGS detection prediction.
