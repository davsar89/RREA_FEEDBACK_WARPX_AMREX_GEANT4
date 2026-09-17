// Maxwell TM RZ smoke: physics checks against exact limits, no reference
// runs.
//
//   LIGHT CONE   a compact Jz ring pulse at the center reaches a probe at
//                distance d no earlier than d/c and has clearly arrived by
//                d/c + pulse width (front speed = c on the grid);
//   RELAXATION   with uniform sigma and a uniform initial Ez, the field
//                decays as exp(-sigma t / eps0) -- the semi-implicit update
//                must track the analytic curve to < 0.5% over 3 tau;
//   ABSORBER     after the pulse leaves, residual field energy is a small
//                fraction of peak (first-order Silver-Mueller, measured);
//   ENERGY       in a closed domain the leapfrog conserves the mixed-time
//                discrete energy exactly, and with sources and the absorber
//                the Ampere step closes dU_E + dU_B + dt(Joule + Poynting)
//                = 0 to round-off: the dual-volume metric and the exact
//                discrete boundary term are what the ledger reports.

#include "rrea/RreaMaxwellTMRZRecord.H"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, char const* what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

rrea::MaxwellTMRZ make_mesh(int nr, int nz, double dr, double dz)
{
    rrea::MaxwellTMRZ f;
    f.nr = nr;
    f.nz = nz;
    f.dr = dr;
    f.dz = dz;
    f.resize();
    return f;
}

// Deterministic O(1) noise in [-1, 1) for field fixtures.
struct Noise {
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    double operator()()
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) / 4503599627370496.0 - 1.0;
    }
};

// Er/Ez as the compatible Yee-face gradient of one smooth potential: the
// discrete curl of such a field is round-off zero (the production ambient
// is built the same way).
rrea::MaxwellTMRZ gradient_field(int nr, int nz, double dr, double dz)
{
    auto f = make_mesh(nr, nz, dr, dz);
    std::vector<double> phi(static_cast<std::size_t>(f.nr) * f.nz, 0.0);
    auto p = [&](int i, int j) -> double& {
        return phi[static_cast<std::size_t>(i)
                   + static_cast<std::size_t>(f.nr) * j];
    };
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 0; i < f.nr; ++i) {
            double const r = (i + 0.5) * f.dr;
            double const z = (j + 0.5) * f.dz;
            p(i, j) = 0.013 * r * r + 0.021 * r * z
                - 0.017 * z * z + 3.0 * std::sin(0.01 * r);
        }
    }
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 1; i < f.nr; ++i) {
            f.Er(i, j) = -(p(i, j) - p(i - 1, j)) / f.dr;
        }
    }
    for (int j = 1; j < f.nz; ++j) {
        for (int i = 0; i < f.nr; ++i) {
            f.Ez(i, j) = -(p(i, j) - p(i, j - 1)) / f.dz;
        }
    }
    return f;
}

double electric_energy(
    rrea::MaxwellTMRZ const& f, rrea::MaxwellTMRZ const* ambient)
{
    rrea::MaxwellTMRZ e = f;
    std::fill(e.btheta.begin(), e.btheta.end(), 0.0);
    return rrea::MaxwellFieldEnergy(e, ambient);
}

void light_cone_and_absorber()
{
    auto f = make_mesh(120, 240, 5.0, 5.0);
    double const dt = 2.5e-9;
    check(f.CflNumber(dt) < 1.0, "CFL below 1");

    // Driven Jz pulse near the axis at mid-height, Gaussian in r and z,
    // BIPOLAR in time (Gaussian derivative): zero net charge transfer, so
    // no static remnant survives -- everything must radiate and be
    // absorbed.  Turn-on sits deep in the tail (exp(-13) of peak) so the
    // light-cone probe is not polluted by a switch-on precursor.
    int const jc = f.nz / 2;
    double const t_peak = 72 * dt;
    double const t_width = 14 * dt;
    // Probe on the same z row, 100 cells = 500 m out in r.  Flight distance
    // is measured from the source's 6-sigma edge (amplitude ~1.5e-8): the
    // source tail is part of the source, not a causality violation.
    int const ip = 100;
    double const src_w = 3.0 * f.dr;
    double const d = ip * f.dr - 6.0 * src_w;
    double const t_front = d / rrea::kMaxwellC;

    std::vector<double> jz(f.ez.size(), 0.0);
    double peak_energy = 0.0;
    double early_max = 0.0;
    double late_max = 0.0;
    int const n_steps = 1600;
    for (int n = 0; n < n_steps; ++n) {
        double const t = n * dt;
        double const gt = -((t - t_peak) / t_width)
            * std::exp(-0.5 * std::pow((t - t_peak) / t_width, 2));
        for (int j = 0; j <= f.nz; ++j) {
            for (int i = 0; i < f.nr; ++i) {
                double const r = (i + 0.5) * f.dr;
                double const z = (j - jc) * f.dz;
                jz[f.EzIndex(i, j)] = 1.0e-6 * gt
                    * std::exp(-0.5 * std::pow(r / src_w, 2))
                    * std::exp(-0.5 * std::pow(z / (4.0 * f.dz), 2));
            }
        }
        rrea::MaxwellStep(f, dt, nullptr, nullptr, nullptr, &jz,
                          nullptr, nullptr);
        double const e_probe = std::fabs(f.Ez(ip, jc));
        if (t < 0.95 * t_front) {
            early_max = std::fmax(early_max, e_probe);
        }
        if (t > t_front && t < t_front + t_peak + 4.0 * t_width) {
            late_max = std::fmax(late_max, e_probe);
        }
        peak_energy = std::fmax(peak_energy, rrea::MaxwellFieldEnergy(f));
    }
    check(late_max > 0.0, "pulse reaches the probe");
    check(early_max <= 3.0e-3 * late_max,
          "probe quiet before the light cone (front at c)");
    double const residual = rrea::MaxwellFieldEnergy(f) / peak_energy;
    std::printf("light cone: early %.3e vs late %.3e; absorber residual "
                "%.2f%% of peak\n", early_max, late_max, 100.0 * residual);
    check(residual < 0.05, "Silver-Mueller absorbs (< 5% residual energy)");
}

void conductive_relaxation()
{
    auto f = make_mesh(24, 48, 5.0, 5.0);
    double const sigma0 = 5.0e-6;             // S/m, collapse-peak scale
    double const tau = rrea::kMaxwellEps0 / sigma0;   // ~1.77 us
    double const dt = 2.5e-9;                 // a = sigma dt/2eps0 ~ 7e-4
    std::vector<double> sigma_r(f.er.size(), sigma0);
    std::vector<double> sigma_z(f.ez.size(), sigma0);

    // Uniform initial Ez; no curl develops from a uniform interior, so with
    // closed boundaries (no Silver-Mueller -- an OPEN uniform field also
    // drains radiatively, which is physics but not what this isolates) the
    // evolution is the pure semi-implicit conductive decay.
    for (double& value : f.ez) { value = 1.0e5; }
    int const ic = f.nr / 2;
    int const jc = f.nz / 2;
    double worst = 0.0;
    int const n_steps = 2400;                 // ~3.4 tau
    for (int n = 1; n <= n_steps; ++n) {
        rrea::MaxwellAdvanceB(f, dt);
        rrea::MaxwellAdvanceE(f, dt, &sigma_r, &sigma_z, nullptr, nullptr,
                              nullptr, nullptr);
        double const analytic = 1.0e5 * std::exp(-n * dt / tau);
        double const rel =
            std::fabs(f.Ez(ic, jc) - analytic) / std::fmax(analytic, 1.0);
        worst = std::fmax(worst, rel);
    }
    std::printf("relaxation: worst relative error vs exp(-t/tau) over "
                "3.4 tau = %.3e\n", worst);
    check(worst < 5.0e-3, "semi-implicit decay tracks eps0/sigma to <0.5%");
}

void compatible_gradient_has_zero_curl()
{
    auto f = gradient_field(17, 23, 5.0, 7.0);
    rrea::MaxwellAdvanceB(f, 1.0);
    double worst = 0.0;
    double scale = 0.0;
    for (int j = 1; j < f.nz; ++j) {
        for (int i = 1; i < f.nr; ++i) {
            worst = std::max(worst, std::abs(f.Bt(i, j)));
            scale = std::max(scale,
                std::abs((f.Ez(i, j) - f.Ez(i - 1, j)) / f.dr)
                + std::abs((f.Er(i, j) - f.Er(i, j - 1)) / f.dz));
        }
    }
    check(worst <= 2.0e-13 * std::max(scale, 1.0),
          "compatible face gradient has zero discrete curl");
}

void divergence_of_curl_is_zero()
{
    auto f = make_mesh(19, 27, 5.0, 6.0);
    for (int j = 0; j <= f.nz; ++j) {
        for (int i = 1; i <= f.nr; ++i) {
            double const r = i * f.dr;
            f.Bt(i, j) = 2.0e-11 * r
                * (0.7 * std::sin(0.23 * j) + 0.2 * std::cos(0.19 * i));
        }
        f.Bt(0, j) = 0.0;
    }
    double const dt = 2.5e-9;
    rrea::MaxwellAdvanceE(
        f, dt, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    double worst = 0.0;
    double scale = 0.0;
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 0; i < f.nr; ++i) {
            worst = std::max(
                worst, std::abs(f.DivergenceRZ(f.er, f.ez, i, j)));
            scale = std::max(scale,
                (std::abs((i + 1) * f.Er(i + 1, j)) + std::abs(i * f.Er(i, j)))
                    / ((i + 0.5) * f.dr)
                + (std::abs(f.Ez(i, j + 1)) + std::abs(f.Ez(i, j))) / f.dz);
        }
    }
    check(worst <= 2.0e-12 * std::max(scale, 1.0),
          "discrete divergence of the Maxwell curl is zero");
}

void axis_radial_curl_limit()
{
    auto f = make_mesh(12, 9, 5.0, 5.0);
    double const slope = 2.0e-12;  // Btheta = slope * r
    for (int j = 0; j <= f.nz; ++j) {
        for (int i = 0; i <= f.nr; ++i) {
            f.Bt(i, j) = slope * i * f.dr;
        }
    }
    double const dt = 2.5e-9;
    rrea::MaxwellAdvanceE(
        f, dt, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    double const expected = dt / (rrea::kMaxwellEps0 * rrea::kMaxwellMu0)
        * 2.0 * slope;
    double worst = 0.0;
    for (double const value : f.ez) {
        worst = std::max(worst, std::abs(value - expected));
    }
    check(worst <= 2.0e-13 * std::max(std::abs(expected), 1.0),
          "axis (1/r)d(rBtheta)/dr limit equals 2 dBtheta/dr");
}

void gauss_constraint_follows_continuity_current()
{
    auto f = make_mesh(15, 18, 5.0, 7.0);
    std::vector<double> jr(f.er.size(), 0.0);
    std::vector<double> jz(f.ez.size(), 0.0);
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 1; i <= f.nr; ++i) {
            jr[f.ErIndex(i, j)] = 3.0e-7 * std::sin(0.13 * i + 0.29 * j);
        }
    }
    for (int j = 0; j <= f.nz; ++j) {
        for (int i = 0; i < f.nr; ++i) {
            jz[f.EzIndex(i, j)] = 2.0e-7 * std::cos(0.17 * i - 0.31 * j);
        }
    }
    double const dt = 2.5e-9;
    rrea::MaxwellAdvanceE(
        f, dt, nullptr, nullptr, &jr, &jz, nullptr, nullptr);
    double worst = 0.0;
    double scale = 0.0;
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 0; i < f.nr; ++i) {
            double const rho_new = -dt * f.DivergenceRZ(jr, jz, i, j);
            double const residual = rrea::kMaxwellEps0
                * f.DivergenceRZ(f.er, f.ez, i, j) - rho_new;
            worst = std::max(worst, std::abs(residual));
            scale = std::max(scale, std::abs(rho_new));
        }
    }
    check(worst <= 5.0e-13 * std::max(scale, 1.0e-30),
          "continuity-consistent current preserves discrete Gauss law");
}

void huygens_surface_is_inset_from_the_absorber()
{
    // The recorded surface must sit strictly inside the domain: on the
    // boundary Silver-Mueller makes Btheta an algebraic copy of the
    // tangential E (no independent magnetic information), which the
    // charge-vs-current cancellation in the radio integral then amplifies.
    check(rrea::kRreaEmSurfaceMarginCells >= 2,
          "EM Huygens surface is inset from the absorbing boundary");

    auto f = make_mesh(40, 40, 5.0, 5.0);
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 0; i <= f.nr; ++i) {
            f.Er(i, j) = 3.0 + 0.25 * ((j + 0.5) * f.dz);   // linear in z
        }
    }
    int const j_face = f.nz - rrea::kRreaEmSurfaceMarginCells;
    double const interpolated =
        0.5 * (f.Er(7, j_face - 1) + f.Er(7, j_face));
    double const exact = 3.0 + 0.25 * (j_face * f.dz);
    check(std::abs(interpolated - exact) < 1.0e-12,
          "inset surface gets tangential E by exact interpolation");
}

void ambient_relative_energy_includes_cross_term()
{
    auto scattered = make_mesh(5, 7, 5.0, 7.0);
    auto ambient = scattered;
    for (std::size_t i = 0; i < scattered.er.size(); ++i) {
        ambient.er[i] = 2.0e4;
        scattered.er[i] = -0.57 * ambient.er[i];
    }
    for (std::size_t i = 0; i < scattered.ez.size(); ++i) {
        ambient.ez[i] = -1.0e5;
        scattered.ez[i] = -0.57 * ambient.ez[i];
    }
    auto total = scattered;
    for (std::size_t i = 0; i < total.er.size(); ++i) {
        total.er[i] += ambient.er[i];
    }
    for (std::size_t i = 0; i < total.ez.size(); ++i) {
        total.ez[i] += ambient.ez[i];
    }
    double const got = rrea::MaxwellFieldEnergy(scattered, &ambient);
    double const reference = rrea::MaxwellFieldEnergy(total)
        - rrea::MaxwellFieldEnergy(ambient);
    check(std::abs(got - reference)
              <= 2.0e-13 * std::max(std::abs(reference), 1.0),
          "ambient-relative energy equals U(total)-U(ambient)");
    check(got < 0.0,
          "screening lowers physical energy despite positive scattered energy");
}

// Closed lossless domain (boundary corners held at zero, no absorber, no
// current): the leapfrog conserves
//   U^n = eps0/2 sum V_f (E^n)^2 + 1/(2 mu0) sum V_c B^{n-1/2} B^{n+1/2}
// exactly.  Any error in a dual volume, the axis terms or the r-weighting
// of the two curls breaks this at O(1), not at round-off.
void closed_domain_energy_is_conserved()
{
    auto f = make_mesh(12, 16, 5.0, 7.0);
    Noise noise;
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 1; i <= f.nr; ++i) { f.Er(i, j) = 1.0e3 * noise(); }
    }
    for (double& value : f.ez) { value = 1.0e3 * noise(); }
    for (int j = 1; j < f.nz; ++j) {
        for (int i = 1; i < f.nr; ++i) { f.Bt(i, j) = 3.0e-6 * noise(); }
    }
    double const dt = 2.5e-9;
    double reference = 0.0;
    double worst = 0.0;
    for (int n = 0; n < 200; ++n) {
        std::vector<double> const b_previous = f.btheta;
        rrea::MaxwellAdvanceB(f, dt);
        double const energy = electric_energy(f, nullptr)
            + rrea::MaxwellMixedMagneticEnergy(f, b_previous);
        if (n == 0) { reference = energy; }
        worst = std::max(worst, std::abs(energy - reference));
        rrea::MaxwellAdvanceE(
            f, dt, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    std::printf("closed domain: mixed-time energy drift %.3e of %.6e J over "
                "200 steps\n", worst / reference, reference);
    check(worst <= 1.0e-12 * std::abs(reference),
          "closed domain conserves the mixed-time discrete energy exactly");
}

// One Ampere substep with sources, the absorber and a curl-free ambient:
//   dU_E(ambient-relative) + dU_B(mixed) + dt (Joule + Poynting) = 0
// with Joule and the boundary term taken as the mean over E^n and E^{n+1}
// at the fixed half-step B, exactly as the production ledger forms them.
void joule_and_poynting_close_the_open_step()
{
    auto ambient = gradient_field(10, 14, 5.0, 7.0);
    auto f = make_mesh(10, 14, 5.0, 7.0);
    Noise noise;
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 1; i <= f.nr; ++i) { f.Er(i, j) = 1.0e3 * noise(); }
    }
    for (double& value : f.ez) { value = 1.0e3 * noise(); }
    for (int j = 0; j <= f.nz; ++j) {
        for (int i = 1; i <= f.nr; ++i) { f.Bt(i, j) = 3.0e-6 * noise(); }
    }
    std::vector<double> jr(f.er.size(), 0.0);
    std::vector<double> jz(f.ez.size(), 0.0);
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 1; i <= f.nr; ++i) { jr[f.ErIndex(i, j)] = 0.1 * noise(); }
    }
    for (double& value : jz) { value = 0.1 * noise(); }

    double const dt = 2.5e-9;
    std::vector<double> const b_before = f.btheta;
    rrea::MaxwellAdvanceB(f, dt);
    rrea::MaxwellSilverMueller(f);
    double const u_e0 = electric_energy(f, &ambient);
    double const u_b0 = rrea::MaxwellMixedMagneticEnergy(f, b_before);
    double const p0 = rrea::MaxwellBoundaryPoyntingPower(f, &ambient);
    double const j0 = rrea::MaxwellJoulePower(f, &ambient, jr, jz);
    rrea::MaxwellAdvanceE(f, dt, nullptr, nullptr, &jr, &jz, nullptr, nullptr);
    double const u_e1 = electric_energy(f, &ambient);
    double const p1 = rrea::MaxwellBoundaryPoyntingPower(f, &ambient);
    double const j1 = rrea::MaxwellJoulePower(f, &ambient, jr, jz);
    auto after = f;
    rrea::MaxwellAdvanceB(after, dt);   // B^{n+3/2} at the interior corners
    double const u_b1 = rrea::MaxwellMixedMagneticEnergy(after, f.btheta);

    double const joule = 0.5 * dt * (j0 + j1);
    double const poynting = 0.5 * dt * (p0 + p1);
    double const residual = (u_e1 - u_e0) + (u_b1 - u_b0) + joule + poynting;
    double const scale = std::abs(u_e1 - u_e0) + std::abs(u_b1 - u_b0)
        + std::abs(joule) + std::abs(poynting);
    std::printf("open step: dU_E %.3e dU_B %.3e Joule %.3e Poynting %.3e "
                "-> residual %.3e of scale\n",
                u_e1 - u_e0, u_b1 - u_b0, joule, poynting, residual / scale);
    check(std::abs(joule) > 0.0 && std::abs(poynting) > 0.0,
          "open-step fixture exercises both the Joule and boundary terms");
    check(std::abs(residual) <= 1.0e-12 * scale,
          "Ampere substep closes dU_E + dU_B + dt(Joule + Poynting) exactly");
}

void boundary_record_restart_recovers_complete_history(bool legacy)
{
    auto field = make_mesh(8, 8, 1.0, 2.0);
    auto const directory = std::filesystem::temp_directory_path()
        / ("rrea_em_record_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    auto const path = directory / "rrea_em_boundary.bin";
    double const times[] = {0.0, 1e-8, 4e-8, 1e-8, 3e-8};
    for (int index = 0; index < 5; ++index) {
        std::fill(field.er.begin(), field.er.end(), 10.0 * (index + 1));
        rrea::AppendMaxwellBoundaryRecord(
            field, directory.string(), times[index] + (legacy ? 1e-8 : 0.0), 1e-8);
    }
    if (legacy) {
        std::fstream record(path, std::ios::in | std::ios::out | std::ios::binary);
        std::int64_t const version = 3;
        record.seekp(8);
        record.write(reinterpret_cast<char const*>(&version), sizeof(version));
    }
    { std::ofstream tail(path, std::ios::binary | std::ios::app); tail.write("bad", 3); }
    bool rejected = false;
    try { rrea::AppendMaxwellBoundaryRecord(field, directory.string(), 2e-8, 1e-8); }
    catch (std::runtime_error const&) { rejected = true; }
    check(rejected, "unaligned EM append requires restart recovery");
    std::fill(field.er.begin(), field.er.end(), 60.0);
    rrea::AppendMaxwellBoundaryRecord(field, directory.string(), 2e-8, 1e-8, true);
    std::ifstream record(path, std::ios::binary);
    std::int64_t version = 0;
    record.seekg(8);
    record.read(reinterpret_cast<char*>(&version), sizeof(version));
    check(version == 4, "EM restart writes the physical E-time convention");
    record.seekg(56);
    double const expected_times[] = {0.0, 1e-8, 1e-8, 2e-8};
    double const expected_er[] = {10.0, 20.0, 40.0, 60.0};
    for (int index = 0; index < 4; ++index) {
        double frame[55] = {};
        record.read(reinterpret_cast<char*>(frame), sizeof(frame));
        check(record.good() && frame[0] == expected_times[index]
              && frame[1] == expected_er[index],
              "EM recovery preserves complete frames and replacement values");
    }
    record.close();
    check(std::filesystem::file_size(path) == 56 + 4 * 55 * 8,
          "EM recovery removes future records and the torn tail");
    field.dr = 2.0;
    rejected = false;
    try { rrea::AppendMaxwellBoundaryRecord(field, directory.string(), 3e-8, 1e-8, true); }
    catch (std::runtime_error const&) { rejected = true; }
    check(rejected, "EM restart rejects changed physical surface geometry");
    std::filesystem::remove(path);
    std::filesystem::remove(directory);
}

}  // namespace

int main()
{
    boundary_record_restart_recovers_complete_history(false);
    boundary_record_restart_recovers_complete_history(true);
    light_cone_and_absorber();
    conductive_relaxation();
    compatible_gradient_has_zero_curl();
    divergence_of_curl_is_zero();
    axis_radial_curl_limit();
    gauss_constraint_follows_continuity_current();
    huygens_surface_is_inset_from_the_absorber();
    ambient_relative_energy_includes_cross_term();
    closed_domain_energy_is_conserved();
    joule_and_poynting_close_the_open_step();
    if (failures) {
        std::fprintf(stderr, "rrea_maxwell_tm_rz_smoke: %d FAILURES\n",
                     failures);
        return 1;
    }
    std::printf("rrea_maxwell_tm_rz_smoke: PASS (light cone, absorber, "
                "relaxation, Yee identities, axis, Gauss, boundary sampling, "
                "ambient-relative energy, exact energy ledger)\n");
    return 0;
}
