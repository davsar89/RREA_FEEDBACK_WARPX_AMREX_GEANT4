"""Cloud/atmosphere radiative transfer for the optical bands, via cloudscat.

The optical source rates from compute_rrea_optical_emissions.py carry their
recorded emission altitude inside the assumed cloud. Therefore
source/(4 pi d^2) is not the transported irradiance: the direct beam is
attenuated and a scattered tail arrives late.  This module hands the source
series to cloudscat_cpp (Luque et al., Geosci. Model Dev. 13, 5549, 2020) and
returns the model observer flux.

Coupling.  cloudscat's `source.type: "list"` takes sub-sources
{position, energy, time}: a photon is drawn from sub-source i with probability
Q_i/sum(Q) and weight 1, isotropically, at r_i and t_i (src/sources.cpp), and
the estimator is divided by N (src/engine.cpp), so the emitted timeline.csv is
photons m^-2 s^-1 PER EMITTED SOURCE PHOTON.  Feeding our own series in makes
the whole coupling one scaling:

    Q_i         = rate(t_i) * dt_i        photons emitted in bin i
    position_i  = [0, 0, alt_m(t_i)]      the series' own emission altitude
    observed(t) = timeline(t) * sum(Q_i)  ph m^-2 s^-1 at the observer

`energy` is only a relative sampling weight (cloudscat normalises it), so the
absolute scale enters solely through sum(Q_i).  Three properties that follow,
each read out of the cloudscat source rather than assumed:

* The DIRECT (unscattered) term is included -- cloudscat next-event estimates
  the initial isotropic emission through exp(-tau) (src/runner.cpp).  This
  therefore REPLACES source/(4 pi d^2); the two must never be added.
* Time is absolute from t = 0 and includes light travel (tobs = t + s_obs/c),
  and the list source applies no extra delay, so the direct pulse arrives at
  d/c, not at t = 0.  The returned axis is on that same
  absolute clock as the input series.
* Passing the series directly assumes no linearity and no fixed source
  position: the sub-source altitudes follow the beam as it drifts, which a
  single impulse-response kernel could not do without an altitude grid.

cloudscat lives in external/cloudscat_cpp as a pristine untracked clone. The
binary is an ELF executable built under WSL, and on Windows it is invoked
through `wsl`.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from rrea_run_support import C

REPO_ROOT = Path(__file__).resolve().parent.parent
CLOUDSCAT_DIR = REPO_ROOT / "external" / "cloudscat_cpp"
CLOUDSCAT_BIN = CLOUDSCAT_DIR / "build" / "cloudscat_run"
CACHE_DIR = REPO_ROOT / "tmp" / "cloudscat_runs"

CLONE_HINT = (
    f"git clone https://git.app.uib.no/aloft/cloudscat_cpp.git "
    f"{CLOUDSCAT_DIR}")
BUILD_HINT = (
    f"wsl cmake -S {CLOUDSCAT_DIR} -B {CLOUDSCAT_DIR}/build "
    f"-DCMAKE_BUILD_TYPE=Release -DCLOUDSCAT_BUILD_TESTS=OFF && "
    f"wsl cmake --build {CLOUDSCAT_DIR}/build -j --target cloudscat_run")

# Cloud properties follow cloudscat's doc/configs/aloft_337.json except for the
# 16 km top used for this intracloud scenario. The source altitude is measured
# from each capture; the top dominates the result, so --cloud-top-km exposes it
# and every figure states it.
ALOFT_CLOUD = {
    "bottom_m": 7000.0,
    "top_m": 16000.0,
    "radius_m": 200000.0,
    "droplet_n_cm3": 100.0,
    "droplet_radius_m": 10e-6,
    "refindex": "Hale.dat",
}
DOMAIN_TOP_M = 60000.0
BAND_WAVELENGTH_M = {"337": 337.1e-9, "777": 777.4e-9}
# The project's standing observer: the ER-2 aircraft at 20 km MSL.  Single
# owner -- every figure/render default imports this rather than repeating the
# literal.
OBSERVER_ALTITUDE_M_MSL = 20000.0


def _wsl_path(path: Path) -> str:
    """Windows path -> /mnt/<drive>/... so the WSL binary can read it."""
    text = str(Path(path).resolve()).replace("\\", "/")
    if len(text) > 1 and text[1] == ":":
        return f"/mnt/{text[0].lower()}{text[2:]}"
    return text


def _command(config_path: Path, out_dir: Path) -> list[str]:
    binary, cfg, out = CLOUDSCAT_BIN, config_path, out_dir
    if platform.system() == "Windows":
        return ["wsl", "-e", _wsl_path(binary), _wsl_path(cfg), _wsl_path(out)]
    return [str(binary), str(cfg), str(out)]


def ensure_cloudscat(build: bool = True) -> None:
    """Require a readable clone and build; physics is checked by self-tests."""
    if not CLOUDSCAT_DIR.exists():
        raise SystemExit(
            f"cloudscat is not cloned at {CLOUDSCAT_DIR}\n  {CLONE_HINT}")
    if CLOUDSCAT_BIN.exists():
        return
    if not build:
        raise SystemExit(f"cloudscat_run is not built\n  {BUILD_HINT}")
    print(f"building cloudscat_run (one time) in {CLOUDSCAT_DIR / 'build'}")
    for stage in (
        ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
         "-DCLOUDSCAT_BUILD_TESTS=OFF"],
        ["cmake", "--build", "build", "-j", "--target", "cloudscat_run"],
    ):
        joined = " ".join(stage)
        cmd = (["wsl", "-e", "bash", "-lc",
                f"cd {_wsl_path(CLOUDSCAT_DIR)} && {joined}"]
               if platform.system() == "Windows"
               else stage)
        result = subprocess.run(
            cmd, cwd=None if platform.system() == "Windows" else CLOUDSCAT_DIR,
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"cloudscat build failed at `{joined}`:\n{result.stderr[-2000:]}"
                f"\nBuild it by hand with:\n  {BUILD_HINT}")
    if not CLOUDSCAT_BIN.exists():
        raise SystemExit(f"cloudscat build reported success but no binary\n  {BUILD_HINT}")


def _trapezoid_node_weights(t: np.ndarray) -> np.ndarray:
    """Trapezoid-rule node weights: half intervals at both endpoints."""
    dt = np.diff(t)
    w = np.empty_like(t)
    w[0] = 0.5 * dt[0]
    w[-1] = 0.5 * dt[-1]
    w[1:-1] = 0.5 * (t[2:] - t[:-2])
    return w


def emission_weighted_altitude(alt_m, weights) -> float:
    """Emission-rate-weighted source altitude [m MSL].

    Single owner: compute_rrea_optical_emissions.observer_geometry and
    scattered_flux both use this, so "how high is the source" cannot
    drift between the optical, EFCM and cloud-transport paths.
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    alt = np.asarray(alt_m, dtype=float)
    return (float(np.average(alt, weights=w))
            if float(np.sum(w)) > 0.0 else float(np.mean(alt)))


def build_config(
    t_s: np.ndarray,
    rate_ph_per_s: np.ndarray,
    alt_m: np.ndarray,
    *,
    band: str,
    n_photons: int = 1_000_000,
    seed: int = 20200,
    observer_alt_m: float = OBSERVER_ALTITUDE_M_MSL,
    tsample_s: float = 2e-6,
    nsamples: int = 2000,
    cloud: dict | None = None,
    rayleigh: bool = True,
    threads: int = 0,
) -> tuple[dict, float]:
    """Return (cloudscat config, total emitted photons sum(Q)).

    cloud=None uses the ALOFT campaign cloud; pass droplet_n_cm3=0 with
    rayleigh=False for the clear-air, direct-only reference.
    """
    if band not in BAND_WAVELENGTH_M:
        raise ValueError(f"band must be one of {sorted(BAND_WAVELENGTH_M)}")
    t_s = np.asarray(t_s, dtype=float)
    rate = np.clip(np.asarray(rate_ph_per_s, dtype=float), 0.0, None)
    alt = np.asarray(alt_m, dtype=float)
    if not (t_s.shape == rate.shape == alt.shape) or t_s.size < 2:
        raise ValueError("t_s, rate_ph_per_s and alt_m must share a shape of >= 2")
    if (not np.all(np.isfinite(t_s))
            or not np.all(np.diff(t_s) > 0.0)):
        raise ValueError("t_s must be finite and strictly increasing")
    if not np.all(np.isfinite(rate_ph_per_s)):
        raise ValueError("rate_ph_per_s must be finite")

    # Photons per sample: trapezoid node weights, so sum(Q) is the integral
    # of the rate.  (np.gradient matched the trapezoid rule only in the
    # interior; its one-sided endpoints doubled the first/last weights,
    # overcounting sum(Q) by 1/(N-1) on a uniform grid.)
    if not np.all(np.isfinite(alt)):
        raise SystemExit("alt_m is not finite; nothing to transport")
    weights = _trapezoid_node_weights(t_s)
    q = rate * weights
    total_photons = float(np.sum(q))
    if not (total_photons > 0.0):
        raise SystemExit(
            "source rate integrates to zero (or is non-finite); "
            "nothing to transport")

    # Empty bins would only waste sampling probability of exactly zero, but
    # they still cost a binary search per photon -- drop them.
    keep = q > 0.0
    cfg_cloud = dict(ALOFT_CLOUD if cloud is None else {**ALOFT_CLOUD, **cloud})
    params = {
        "N": int(n_photons),
        "lambda": BAND_WAVELENGTH_M[band],
        "seed": int(seed),
    }
    if not rayleigh:
        params["sigma_ray"] = 0
    config = {
        "params": params,
        "source": {
            "type": "list",
            "points": [
                {"position": [0.0, 0.0, float(z)], "energy": float(w),
                 "time": float(t)}
                for t, w, z in zip(t_s[keep], q[keep], alt[keep], strict=True)
            ],
        },
        "cloud": {
            "type": "cylinder",
            "bottom": cfg_cloud["bottom_m"],
            "top": cfg_cloud["top_m"],
            "r": cfg_cloud["radius_m"],
        },
        "domain": {
            "type": "cylinder",
            "bottom": cfg_cloud["bottom_m"],
            "top": DOMAIN_TOP_M,
            "r": cfg_cloud["radius_m"],
        },
        "composition": {
            "type": "homogeneous",
            "n_cm3": cfg_cloud["droplet_n_cm3"],
            "radius": cfg_cloud["droplet_radius_m"],
            "refindex": cfg_cloud["refindex"],
        },
        "observers": [{
            "position": [0.0, 0.0, float(observer_alt_m)],
            "tsample": float(tsample_s),
            "nsamples": int(nsamples),
            "fov": 80,
            "pixels": 8,
        }],
        "depth": False,
        "threads": int(threads),
    }
    return config, total_photons


def run_cached(config: dict, *, cache_dir: Path | None = None,
               verbose: bool = True) -> Path:
    """Run cloudscat for this config, or reuse an identical earlier run.

    Reuses a prior case only when its stored parsed configuration is identical.
    """
    ensure_cloudscat()
    root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("case_*"))
    for candidate in existing:
        config_path = candidate / "config.json"
        timeline = candidate / "obs00001" / "timeline.csv"
        if config_path.is_file() and timeline.is_file() \
                and config_path.read_text(encoding="utf-8") == blob:
            if verbose:
                print(f"cloudscat: reusing cached run {candidate}")
            return candidate
    used = [int(path.name.split("_")[-1]) for path in existing
            if path.name.split("_")[-1].isdigit()]
    out_dir = root / f"case_{max(used, default=0) + 1:04d}"
    timeline = out_dir / "obs00001" / "timeline.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(blob, encoding="utf-8")
    if verbose:
        print(f"cloudscat: transporting {config['params']['N']:,} photons "
              f"at {config['params']['lambda']*1e9:.1f} nm "
              f"({len(config['source']['points'])} sub-sources)")
    result = subprocess.run(_command(config_path, out_dir),
                            capture_output=True, text=True, check=False)
    if result.returncode != 0 or not timeline.exists():
        raise SystemExit(
            f"cloudscat run failed (exit {result.returncode}):\n"
            f"{(result.stderr or result.stdout)[-2000:]}")
    if verbose:
        print(f"cloudscat: {result.stderr.strip().splitlines()[-1]}"
              if result.stderr.strip() else "cloudscat: done")
    return out_dir


def read_timeline(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(t_s, flux per source photon) from a cloudscat observer directory."""
    rows = np.genfromtxt(out_dir / "obs00001" / "timeline.csv",
                         delimiter=",", names=True)
    return np.atleast_1d(rows["t"]), np.atleast_1d(rows["flux"])


def scattered_flux(
    t_s: np.ndarray,
    rate_ph_per_s: np.ndarray,
    alt_m: np.ndarray,
    *,
    band: str,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Observed photon flux [ph m^-2 s^-1] at the observer, with transport.

    Returns (absolute_arrival_time_s, flux, captured).  `captured` is one
    because the native CloudScat observer grid is returned without truncating
    or shifting it.  Every source point keeps its own emission altitude and
    therefore its own light-travel time.
    """
    config, total_photons = build_config(
        t_s, rate_ph_per_s, alt_m, band=band, **kwargs)
    out_dir = run_cached(config)
    t_obs, flux_per_photon = read_timeline(out_dir)
    flux = flux_per_photon * total_photons
    return t_obs, flux, 1.0


def self_test() -> int:
    """Closed-form check of the whole coupling.

    With no droplets and no Rayleigh scattering the only contribution is the
    direct term, whose next-event estimator is 1/(4 pi s^2 N) per photon.  The
    time-integrated observed fluence must then equal sum(Q)/(4 pi d^2) -- the
    same number the geometric panel of compute_rrea_optical_emissions.py
    computes -- which pins units, normalisation, geometry and the sum(Q)
    scaling at once.  The direct pulse must also land at d/c.
    """
    failures = []
    src_alt, obs_alt = 10000.0, 20000.0
    d = obs_alt - src_alt
    t = np.linspace(0.0, 40e-6, 5)
    rate = np.full_like(t, 1e12)
    alt = np.full_like(t, src_alt)

    # Independent quadrature oracle first: for a constant rate the emitted
    # total has the closed form rate*(t_end - t_start), computable with no
    # transport at all.  The direct-only check below CANNOT see a wrong
    # sum(Q) (build_config sits on both of its sides), so this line is the
    # one that pins the time integral itself.
    _, total = build_config(t, rate, alt, band="337")
    expected_q = float(rate[0]) * float(t[-1] - t[0])
    rel_q = abs(total / expected_q - 1.0)
    print(f"{'PASS' if rel_q < 1e-12 else 'FAIL'}  trapezoid sum(Q) = "
          f"{total:.6e} vs closed form {expected_q:.6e} (rel {rel_q:.2e})")
    if rel_q >= 1e-12:
        failures.append("trapezoid quadrature")

    t_obs, flux, _ = scattered_flux(
        t, rate, alt, band="337",
        n_photons=20000, tsample_s=1e-6, nsamples=200,
        cloud={"droplet_n_cm3": 0.0}, rayleigh=False)
    fluence = float(np.sum(flux) * (t_obs[1] - t_obs[0]))
    expected = total / (4.0 * np.pi * d * d)
    rel = abs(fluence / expected - 1.0)
    print(f"{'PASS' if rel < 1e-6 else 'FAIL'}  direct-only fluence = "
          f"{fluence:.6e} vs analytic {expected:.6e} (rel {rel:.2e})")
    if rel >= 1e-6:
        failures.append("direct-only normalisation")

    # The source spans 0-40 us and the light takes d/c = 33.4 us, so first
    # light must arrive at d/c and the last at 40 us + d/c.
    lit = t_obs[flux > 0.0]
    first, last = float(lit[0]), float(lit[-1])
    want_first, want_last = d / C, 40e-6 + d / C
    ok_time = abs(first - want_first) <= 1e-6 and abs(last - want_last) <= 1e-6
    print(f"{'PASS' if ok_time else 'FAIL'}  direct arrival window "
          f"{first*1e6:.1f}-{last*1e6:.1f} us vs expected "
          f"{want_first*1e6:.1f}-{want_last*1e6:.1f} us")
    if not ok_time:
        failures.append("light-travel time origin")

    # The cloudy self-test places the source at high Mie optical depth, so the
    # direct beam is extinguished, not merely dimmed: what the observer sees is
    # multiply-scattered light, delayed well past d/c.  Note there is NO
    # inequality between this and the clear-air fluence -- scattering redirects
    # photons that would otherwise have missed the observer entirely, so the
    # cloudy fluence legitimately exceeds the bare line-of-sight value.
    t_cl, flux_cl, _ = scattered_flux(
        t, rate, alt, band="337",
        n_photons=20000, tsample_s=1e-6, nsamples=2000)
    dt_cl = t_cl[1] - t_cl[0]
    fluence_cl = float(np.sum(flux_cl) * dt_cl)
    direct = float(np.sum(flux_cl[(t_cl >= want_first) & (t_cl <= want_last)])
                   * dt_cl)
    direct_share = direct / max(fluence_cl, 1e-300)
    ok_buried = direct_share < 0.5 and t_cl[int(np.argmax(flux_cl))] > want_first
    print(f"{'PASS' if ok_buried else 'FAIL'}  buried source: direct window "
          f"holds {direct_share:.1%} of the fluence, peak delayed to "
          f"{t_cl[int(np.argmax(flux_cl))]*1e6:.0f} us (direct at "
          f"{want_first*1e6:.0f} us)")
    if not ok_buried:
        failures.append("buried-source diffusion")

    # The sum(Q) scaling must be exactly linear in the source rate: the
    # sampling probabilities are normalised, so scaling every Q by 10 leaves
    # cloudscat's per-photon timeline identical and only moves our factor.
    _, flux_x10, _ = scattered_flux(
        t, rate * 10.0, alt, band="337",
        n_photons=20000, tsample_s=1e-6, nsamples=2000)
    ratio = float(np.max(flux_x10) / np.max(flux_cl))
    ok_linear = abs(ratio - 10.0) < 1e-9
    print(f"{'PASS' if ok_linear else 'FAIL'}  sum(Q) scaling is linear: "
          f"x10 source -> x{ratio:.9f} flux")
    if not ok_linear:
        failures.append("sum(Q) linearity")

    if failures:
        print(f"self-test FAIL ({len(failures)}: {', '.join(failures)})")
        return 1
    print("self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
