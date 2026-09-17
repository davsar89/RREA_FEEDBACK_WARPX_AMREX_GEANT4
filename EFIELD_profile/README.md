# Profiled Atmosphere Inputs

This directory contains the checked-in atmosphere and electric-field profiles
used by production captures and Coleman-Dwyer comparisons.

The simulation coordinate follows `capture_defaults` in
`config/rrea_defaults.json`:

```text
altitude_m = altitude_at_z0_m + z_m
```

The field CSV spans 4.0-16.2 km MSL and is zero above 13.7 km by data (the
loader returns zero outside its range); the density CSV spans 0-20 km. The
capture driver resolves and audits the configured domain against these profiles.

Required CSV files:

- `digitized_efield_profile.csv`: electric-field shape versus altitude. The
  capture driver normalizes it by its peak absolute value and applies the
  requested case value, `E0_peak`.
- `rrea_threshold_NRLMSISE00_gulf_284kVpm.csv`: the absolute neutral
  mass-density profile versus altitude. The density ratio and the scaled RREA
  threshold are DERIVED from it against explicit reference densities -- the
  file's own `*_ratio_to_sea_level` and `RREA_Eth_abs_kV_per_m` columns are
  deliberately not read (see `scripts/rrea_profiled_atmosphere.py`).
