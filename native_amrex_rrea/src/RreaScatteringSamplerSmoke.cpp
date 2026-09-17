#include "rrea/RreaGpuSmoke.H"
// Sampler-level spectrum smoke for the discrete scattering channels.
//
// Loads the requested semantic transport configuration and evaluates each sampler's quantile
// function on a dense deterministic u-grid (the samplers are pure functions
// of (E, u), so this measures the exact sampled distribution with no Monte
// Carlo noise), comparing against independent analytic references:
//
//   - Moller secondary energy vs the closed-form Moller CDF and mean transfer;
//   - Bhabha secondary energy vs the full Bhabha DCS CDF;
//   - brems photon energy endpoints, the 1/k spectral shape of both species,
//     and mean photon fraction vs an independent Geant4 holdout;
//   - bounded ModifiedTsai angle moments, support and complete brems events;
//   - in-flight annihilation final state: energy closure, kinematic support,
//     and mean photon energy fraction vs the Heitler spectrum.
// (The Moller/Bhabha lab-angle kinematics are tested against the engine's own
// functions in rrea_charged_interleave_smoke.)
//
// usage: rrea_scattering_sampler_smoke <transport_physics.json>

#include "rrea/RreaInteractionTables.H"
#include "rrea/RreaTableContainer.H"
#include "RreaChargedFinalStates.H"

#include <AMReX.H>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kMeC2eV = 510998.95069;
// Transport configuration-sourced in main(): the analytic Moller/Bhabha references are built
// at whatever hard-secondary threshold the loaded bundle declares.
double g_hard_secondary_eV = 0.0;
// Fixed outputs pin the deterministic mapping from uniform draws to sampler
// states; they are not physics evidence. Independent Klein-Nishina moments and
// Sauter-Gavrila quadrature below provide the physics anchors.
constexpr double GOLD_C1 = 0.253592432377;  // Compton cos(theta) @ 5 MeV, u=(0.3,0.7)
constexpr double GOLD_C2 = 0.992888749965;  // Compton cos(theta) @ 50 MeV, u=(0.9,0.1)
constexpr double GOLD_P1 = 0.334257091114;  // photoelectron cos(theta) @ 100 keV, u=0.3
constexpr double GOLD_P2 = 0.662755843470;  // photoelectron cos(theta) @ 30 keV, u=0.8
int g_failures = 0;

void check(bool ok, std::string const& label, std::string const& detail)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << ": " << detail << "\n";
    if (!ok) { ++g_failures; }
}

struct Kin {
    double gamma, beta2, tau;
};

Kin kin(double k_eV)
{
    Kin out;
    out.tau = k_eV / kMeC2eV;
    out.gamma = 1.0 + out.tau;
    out.beta2 = 1.0 - 1.0 / (out.gamma * out.gamma);
    return out;
}

// ---- Moller closed-form reference ----------------------------------------

double moller_bracket(double k_eV, double eps)
{
    auto const q = kin(k_eV);
    double const g2 = q.gamma * q.gamma;
    return (q.tau * q.tau / g2) * (0.5 - eps) + 1.0 / eps - 1.0 / (1.0 - eps)
           - ((2.0 * q.gamma - 1.0) / g2) * std::log((1.0 - eps) / eps);
}

double moller_cdf(double k_eV, double t_eV)
{
    double const x0 = g_hard_secondary_eV / k_eV;
    double const x = std::min(std::max(t_eV / k_eV, x0), 0.5);
    double const total = moller_bracket(k_eV, x0);
    if (x >= 0.5) { return 1.0; }
    return (total - moller_bracket(k_eV, x)) / total;
}

double moller_dcs_shape(double k_eV, double eps)
{
    auto const q = kin(k_eV);
    double const g2 = q.gamma * q.gamma;
    return 1.0 / (eps * eps) + 1.0 / ((1.0 - eps) * (1.0 - eps)) + (q.tau * q.tau) / g2
           - ((2.0 * q.gamma - 1.0) / g2) / (eps * (1.0 - eps));
}

double moller_mean_secondary_eV(double k_eV)
{
    double const x0 = g_hard_secondary_eV / k_eV;
    int const n = 20000;
    double const l0 = std::log(x0);
    double const l1 = std::log(0.5);
    double num = 0.0;
    double den = 0.0;
    double prev_eps = x0;
    double prev_f = moller_dcs_shape(k_eV, x0);
    for (int i = 1; i <= n; ++i) {
        double const eps = std::exp(l0 + (l1 - l0) * i / n);
        double const f = moller_dcs_shape(k_eV, eps);
        double const w = 0.5 * (eps - prev_eps);
        num += w * (prev_eps * prev_f + eps * f);
        den += w * (prev_f + f);
        prev_eps = eps;
        prev_f = f;
    }
    return k_eV * num / den;
}

// ---- Full-Bhabha reference CDF --------------------------------------------

struct BhabhaRef {
    std::vector<double> eps;
    std::vector<double> cum;  // normalized CDF over [eps_c, 1]
};

BhabhaRef bhabha_reference(double k_eV)
{
    auto const q = kin(k_eV);
    double const y = 1.0 / (q.gamma + 1.0);
    double const one2y = 1.0 - 2.0 * y;
    double const b1 = 2.0 - y * y;
    double const b2 = one2y * (3.0 + y * y);
    double const b3 = one2y * one2y + one2y * one2y * one2y;
    double const b4 = one2y * one2y * one2y;
    auto dcs = [&](double eps) {
        double const v = 1.0 / (q.beta2 * eps * eps) - b1 / eps + b2 - b3 * eps + b4 * eps * eps;
        return std::max(v, 0.0);
    };
    BhabhaRef ref;
    double const eps_c = g_hard_secondary_eV / k_eV;
    int const n = 4000;
    double const l0 = std::log(eps_c);
    ref.eps.push_back(eps_c);
    ref.cum.push_back(0.0);
    double prev_eps = eps_c;
    double prev_f = dcs(eps_c);
    double total = 0.0;
    for (int i = 1; i <= n; ++i) {
        double const eps = std::exp(l0 + (0.0 - l0) * i / n);
        double const f = dcs(eps);
        total += 0.5 * (eps - prev_eps) * (prev_f + f);
        ref.eps.push_back(eps);
        ref.cum.push_back(total);
        prev_eps = eps;
        prev_f = f;
    }
    for (auto& c : ref.cum) { c /= total; }
    return ref;
}

double bhabha_cdf(BhabhaRef const& ref, double eps)
{
    if (eps <= ref.eps.front()) { return 0.0; }
    if (eps >= ref.eps.back()) { return 1.0; }
    auto const it = std::lower_bound(ref.eps.begin(), ref.eps.end(), eps);
    std::size_t const hi = static_cast<std::size_t>(it - ref.eps.begin());
    std::size_t const lo = hi - 1;
    double const t = (eps - ref.eps[lo]) / (ref.eps[hi] - ref.eps[lo]);
    return ref.cum[lo] + t * (ref.cum[hi] - ref.cum[lo]);
}

// ---- Bounded ModifiedTsai reference moments -------------------------------

void tsai_moments(double gamma, double& mean_u, double& rms_u)
{
    // Independent density quadrature, not the sampler's inverse-CDF formula.
    // u = gamma*sqrt(2*(1-cos(theta))); condition the full density on [0,2*gamma].
    double const a = 0.625;
    double const d = 27.0;
    int const n = 200000;
    double const umax = std::min(2.0 * gamma, 80.0); // omitted tail < 3e-20
    double m0 = 0.0;
    double m1 = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double const u = umax * (i + 0.5) / n;
        double const f = u * std::exp(-a * u) + d * u * std::exp(-3.0 * a * u);
        m0 += f;
        m1 += f * u;
        m2 += f * u * u;
    }
    mean_u = m1 / m0;
    rms_u = std::sqrt(m2 / m0);
}

void exercise_brems_events(rrea::RreaInteractionTables const& tables)
{
    using namespace rrea::warpx;
    for (bool positron : {false, true}) {
        bool valid = true;
        double worst_energy = 0.0, worst_norm = 0.0;
        int exact_backward = 0;
        for (double energy : {56234.13251903491, 1.0e5, 1.0e6}) {
            for (std::uint64_t i = 0; i < 5000; ++i) {
                TransportSideEffects fx;
                fx.SetParentContext(i, positron ? RreaSecondarySpecies::Positron
                                               : RreaSecondarySpecies::Electron);
                int draws = 0;
                auto key = [&](RreaRngChannel channel) {
                    ++draws;
                    return RreaRngKey{8128, i, 1, 0, static_cast<std::uint64_t>(channel)};
                };
                amrex::Real kinetic = energy, dx = 0.0, dy = 0.0, dz = 1.0;
                amrex::Real local = 0.0;
                RreaInteractionSite const site{0.0, 0.0, 1.0, 0.0, 0.25};
                RreaApplyBremsFinalState(fx, tables, key,
                    [&](amrex::Real value) { local += value; }, site, 3.0,
                    positron ? 1.0 : -1.0, tables.BremsAbsoluteCutoffEv(),
                    kinetic, dx, dy, dz);
                valid = valid && draws == 3 && !fx.group_open && fx.groups.size() == 1;
                if (fx.groups.size() != 1 || fx.groups.front().particles.size() != 1) {
                    valid = false;
                    continue;
                }
                auto const& group = fx.groups.front();
                auto const& photon = group.particles.front();
                valid = valid && group.process == RreaSecondaryProcess::Bremsstrahlung
                    && photon.species == RreaSecondarySpecies::Photon && photon.weight == 3.0
                    && photon.birth_elapsed_step_fraction == site.elapsed_step_fraction
                    && kinetic >= 0.0 && local == 0.0
                    && std::isfinite(dx) && std::isfinite(dy) && std::isfinite(dz)
                    && std::isfinite(photon.dir_x) && std::isfinite(photon.dir_y)
                    && std::isfinite(photon.dir_z);
                worst_energy = std::max(worst_energy,
                    std::abs(kinetic + photon.kinetic_or_photon_energy_eV - energy) / energy);
                worst_norm = std::max({worst_norm,
                    std::abs(dx*dx + dy*dy + dz*dz - 1.0),
                    std::abs(photon.dir_x*photon.dir_x + photon.dir_y*photon.dir_y
                        + photon.dir_z*photon.dir_z - 1.0)});
                exact_backward += photon.dir_z == -1.0;
            }
        }
        check(valid && worst_energy < 1.0e-13 && worst_norm < 1.0e-13 && exact_backward == 0,
            positron ? "positron_brems_final_state" : "electron_brems_final_state",
            "15000 shared-path events; relative energy residual=" + std::to_string(worst_energy)
                + ", norm residual=" + std::to_string(worst_norm)
                + ", exactly backward=" + std::to_string(exact_backward));
    }
}

// ---- Heitler annihilation spectrum reference -------------------------------

void heitler_reference(double k_eV, double& eps_lo, double& eps_hi, double& mean_eps)
{
    double const gamma = 1.0 + k_eV / kMeC2eV;
    double const beta_cm = std::sqrt((gamma - 1.0) / (gamma + 1.0));
    eps_lo = 0.5 * (1.0 - beta_cm);
    eps_hi = 0.5 * (1.0 + beta_cm);
    double const gp1 = gamma + 1.0;
    auto f = [&](double eps) {
        return (1.0 / eps)
               * (1.0 + 2.0 * gamma / (gp1 * gp1) - eps - 1.0 / (gp1 * gp1 * eps));
    };
    int const n = 20000;
    double m0 = 0.0;
    double m1 = 0.0;
    for (int i = 0; i < n; ++i) {
        double const eps = eps_lo + (eps_hi - eps_lo) * (i + 0.5) / n;
        double const w = std::max(f(eps), 0.0);
        m0 += w;
        m1 += w * eps;
    }
    mean_eps = m1 / m0;
}

// ---- Klein-Nishina reference moments (dsigma/dmu over mu in [-1,1]) ---------

void kn_moments(double alpha, double& mean_eps, double& mean_mu, double& eps_min)
{
    int const n = 400000;
    double const h = 2.0 / n;
    double m0 = 0.0, meps = 0.0, mmu = 0.0;
    for (int i = 0; i <= n; ++i) {
        double const mu = -1.0 + h * i;
        double const w = (i == 0 || i == n) ? 0.5 : 1.0;
        double const eps = 1.0 / (1.0 + alpha * (1.0 - mu));
        double const sin2 = 1.0 - mu * mu;
        double const f = eps * eps * (eps + 1.0 / eps - sin2);  // KN dsigma/dmu shape
        m0 += w * f;
        meps += w * f * eps;
        mmu += w * f * mu;
    }
    mean_eps = meps / m0;
    mean_mu = mmu / m0;
    eps_min = 1.0 / (1.0 + 2.0 * alpha);
}

// ---- Independent Sauter-Gavrila reference ----------------------------------
//
// Integrate numerically in y=log(t/t_lo), not uniformly in mu.  The latter
// misses the increasingly narrow forward peak above a few MeV.  The y-density
// is smooth across the complete certified 1 keV--10 GeV range and does not use
// the runtime sampler's analytic primitive.

struct SauterGavrilaReference {
    double mean_mu = 0.0;
    double mean_mu2 = 0.0;
    std::vector<double> quantile_mu;
};

SauterGavrilaReference sauter_gavrila_reference(
    double k_eV,
    std::vector<double> const& quantiles)
{
    long double const gamma = 1.0L + static_cast<long double>(k_eV) / kMeC2eV;
    long double const inv_gamma_sq = 1.0L / (gamma * gamma);
    long double const beta = std::sqrt(1.0L - inv_gamma_sq);
    long double const b = 0.5L * gamma * (gamma - 1.0L) * (gamma - 2.0L);
    long double const t_lo = inv_gamma_sq / (1.0L + beta);
    long double const t_hi = 1.0L + beta;
    long double const log_span = std::log(t_hi / t_lo);
    int constexpr n = 200000;
    long double const h = log_span / static_cast<long double>(n);

    std::vector<long double> cumulative(static_cast<std::size_t>(n) + 1U, 0.0L);
    long double total = 0.0L;
    long double first_moment = 0.0L;
    long double second_moment = 0.0L;
    long double previous_w = 0.0L;
    long double previous_w_mu = 0.0L;
    long double previous_w_mu2 = 0.0L;
    for (int i = 0; i <= n; ++i) {
        long double const y = h * static_cast<long double>(i);
        long double const t = t_lo * std::exp(y);
        long double const mu = (1.0L - t) / beta;
        long double const w = std::max(
            0.0L,
            (t - t_lo) * (t_hi - t) * (1.0L + b * t) / (t * t * t));
        long double const w_mu = w * mu;
        long double const w_mu2 = w_mu * mu;
        if (i > 0) {
            total += 0.5L * h * (previous_w + w);
            first_moment += 0.5L * h * (previous_w_mu + w_mu);
            second_moment += 0.5L * h * (previous_w_mu2 + w_mu2);
            cumulative[static_cast<std::size_t>(i)] = total;
        }
        previous_w = w;
        previous_w_mu = w_mu;
        previous_w_mu2 = w_mu2;
    }
    if (!(total > 0.0L) || !std::isfinite(total)) {
        throw std::runtime_error("invalid Sauter-Gavrila numerical reference normalization");
    }

    SauterGavrilaReference out;
    out.mean_mu = static_cast<double>(first_moment / total);
    out.mean_mu2 = static_cast<double>(second_moment / total);
    out.quantile_mu.reserve(quantiles.size());
    for (double quantile : quantiles) {
        // y increases from mu=+1 to mu=-1, hence its cumulative probability is
        // 1-F_mu.  Invert the numerical trapezoid cumulative by interpolation.
        long double const wanted = (1.0L - static_cast<long double>(quantile)) * total;
        auto const it = std::lower_bound(cumulative.begin(), cumulative.end(), wanted);
        std::size_t const upper = static_cast<std::size_t>(it - cumulative.begin());
        long double y = 0.0L;
        if (upper >= cumulative.size()) {
            y = log_span;
        } else if (upper > 0U) {
            long double const below = cumulative[upper - 1U];
            long double const above = cumulative[upper];
            long double const fraction = above > below ? (wanted - below) / (above - below) : 0.0L;
            y = h * (static_cast<long double>(upper - 1U) + fraction);
        }
        long double const t = t_lo * std::exp(y);
        out.quantile_mu.push_back(static_cast<double>((1.0L - t) / beta));
    }
    return out;
}

std::string fmt(double v)
{
    std::ostringstream ss;
    ss.precision(4);
    ss << v;
    return ss.str();
}

}  // namespace

int main(int argc, char** argv)
{
    rrea::smoke::GpuSession gpu_runtime;
    if (argc < 2) {
        std::cerr << "usage: rrea_scattering_sampler_smoke <transport_physics.json>\n";
        return EXIT_FAILURE;
    }
    rrea::RreaInteractionTables tables;
    try {
        tables.Load(argv[1]);
    } catch (std::exception const& exc) {
        std::cerr << "table load failed: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
    g_hard_secondary_eV = static_cast<double>(tables.HardMollerSecondaryThresholdEv());
    std::vector<double> const energies_MeV = {0.3, 1.0, 5.0, 20.0, 100.0};
    int const n_u = 200001;

    // ---- Moller secondary spectrum ----------------------------------------
    for (double e_mev : energies_MeV) {
        double const k_eV = e_mev * 1.0e6;
        double max_du = 0.0;
        double mean_t = 0.0;
        double t_min = 1e300;
        double t_max = -1e300;
        for (int i = 0; i < n_u; ++i) {
            double const u = (i + 0.5) / n_u;
            double const t = tables.SampleMollerSecondaryEnergy(k_eV, u);
            mean_t += t;
            t_min = std::min(t_min, t);
            t_max = std::max(t_max, t);
            max_du = std::max(max_du, std::abs(u - moller_cdf(k_eV, t)));
        }
        mean_t /= n_u;
        double const mean_ref = moller_mean_secondary_eV(k_eV);
        check(max_du <= 0.005, "moller_cdf_distance_" + fmt(e_mev) + "MeV",
              "max|u - u_analytic| = " + fmt(max_du) + " (tol 0.005)");
        check(std::abs(mean_t / mean_ref - 1.0) <= 0.01,
              "moller_mean_secondary_" + fmt(e_mev) + "MeV",
              "sampled <T> = " + fmt(mean_t) + " eV vs analytic " + fmt(mean_ref)
                  + " eV, ratio " + fmt(mean_t / mean_ref) + " (tol 1%)");
        check(t_min >= g_hard_secondary_eV * (1.0 - 1.0e-9) && t_max <= 0.5 * k_eV * (1.0 + 1.0e-9),
              "moller_support_" + fmt(e_mev) + "MeV",
              "T in [" + fmt(t_min) + ", " + fmt(t_max) + "] eV");
    }

    // ---- Bhabha secondary spectrum vs full DCS -----------------------------
    for (double e_mev : energies_MeV) {
        double const k_eV = e_mev * 1.0e6;
        auto const ref = bhabha_reference(k_eV);
        double max_du = 0.0;
        for (int i = 0; i < n_u; ++i) {
            double const u = (i + 0.5) / n_u;
            double const t = tables.SamplePositronBhabhaSecondaryEnergy(k_eV, u);
            max_du = std::max(max_du, std::abs(u - bhabha_cdf(ref, t / k_eV)));
        }
        check(max_du <= 0.01, "bhabha_cdf_vs_full_dcs_" + fmt(e_mev) + "MeV",
              "max|u - u_fullDCS| = " + fmt(max_du) + " (tol 0.01)");
    }

    // ---- Brems photon energy ------------------------------------------------
    // The cut comes from the transport configuration, so this holds for any regenerated bundle.
    for (double e_mev : energies_MeV) {
        double const k_eV = e_mev * 1.0e6;
        double const q0 = tables.SampleBremsstrahlungPhotonEnergy(k_eV, 0.0);
        double const q1 = tables.SampleBremsstrahlungPhotonEnergy(k_eV, 1.0);
        // The brems CDFs are empirical Geant4 quantiles: the endpoints
        // are the smallest/largest OBSERVED photon fractions -- the bottom
        // sits just above the exact tracked cut and the top just below K.
        double const q0_ratio = q0 / tables.BremsAbsoluteCutoffEv();
        double const q1_ratio = q1 / k_eV;
        bool const endpoints_ok =
            q0_ratio >= 1.0 - 1.0e-6 && q0_ratio <= 1.01
            && q1_ratio >= 0.99 && q1_ratio <= 1.0 + 1.0e-6;
        check(endpoints_ok,
              "brems_energy_endpoints_" + fmt(e_mev) + "MeV",
              "q(0)/k_min = " + fmt(q0_ratio) + ", q(1)/K = " + fmt(q1_ratio));
    }

    // ---- Brems photon energy: spectral shape, both species -------------------
    // Endpoints alone cannot see a wrongly shaped interior, and the positron
    // spectrum has no endpoint check at all.
    // Bremsstrahlung is close to a 1/k spectrum, so the CDF of a correct
    // sampler tracks u = ln(k/k_min)/ln(k_max/k_min).  The residual is the
    // Bethe-Heitler shape factor and is real physics, not error: it is largest
    // at low energy and shrinks as the spectrum hardens. Require
    // that ordering as well -- a spectrum wrong at high energy could sit inside
    // a loose envelope but cannot reproduce the trend.  The two species are
    // then compared against each other, which is what actually guards the
    // positron table: its shape must track the electron one it is built beside.
    {
        std::map<double, std::vector<double>> quantiles_by_energy;
        for (bool is_positron : {false, true}) {
            char const* species = is_positron ? "positron" : "electron";
            double previous_du = 1.0;
            for (double e_mev : {1.0, 5.0, 20.0}) {
                double const k_eV = e_mev * 1.0e6;
                double const k_min = tables.BremsAbsoluteCutoffEv();
                double const span = std::log(k_eV / k_min);
                double max_du = 0.0;
                double previous_k = 0.0;
                bool monotone = true;
                bool in_support = true;
                for (int i = 0; i <= 100; ++i) {
                    double const u = i / 100.0;
                    double const k = is_positron
                        ? tables.SamplePositronBremsstrahlungPhotonEnergy(k_eV, u)
                        : tables.SampleBremsstrahlungPhotonEnergy(k_eV, u);
                    if (k < previous_k) { monotone = false; }
                    previous_k = k;
                    if (k < k_min * (1.0 - 1.0e-6) || k > k_eV * (1.0 + 1.0e-6)) {
                        in_support = false;
                    }
                    double const k_clamped = std::min(std::max(k, k_min), k_eV);
                    max_du = std::max(max_du, std::abs(u - std::log(k_clamped / k_min) / span));
                    quantiles_by_energy[e_mev].push_back(k / k_eV);
                }
                std::ostringstream d;
                d.precision(4);
                d << species << " E=" << e_mev << "MeV max|u - u_1/k| = " << max_du
                  << " (previous energy " << previous_du << ") monotone=" << monotone
                  << " in_support=" << in_support;
                check(monotone && in_support && max_du <= 0.25 && max_du < previous_du,
                      "brems photon spectrum approaches 1/k as it hardens", d.str());
                previous_du = max_du;
            }
        }
        for (auto const& [e_mev, kappa] : quantiles_by_energy) {
            std::size_t const half = kappa.size() / 2;
            double max_rel = 0.0;
            for (std::size_t i = 0; i < half; ++i) {
                max_rel = std::max(max_rel,
                                   std::abs(kappa[i] - kappa[half + i])
                                       / std::max(kappa[i], 1.0e-30));
            }
            std::ostringstream d;
            d.precision(4);
            d << "E=" << e_mev << "MeV max relative quantile difference = " << max_rel;
            check(max_rel <= 0.10,
                  "positron and electron brems spectra share a shape", d.str());
        }
    }

    // ---- Bounded brems angles, including mildly relativistic parents --------
    {
        for (double e_mev : {0.001, 0.05623413251903491, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0}) {
            double const k_eV = e_mev * 1.0e6;
            double const gamma_tot = 1.0 + k_eV / kMeC2eV;
            double tsai_mean = 0.0, tsai_rms = 0.0;
            tsai_moments(gamma_tot, tsai_mean, tsai_rms);
            double m1 = 0.0, m2 = 0.0, previous = 1.0;
            bool supported = true, monotone = true;
            int backward = 0;
            constexpr int n_angles = 100000;
            for (int i = 0; i < n_angles; ++i) {
                double const u = (i + 0.5) / n_angles;
                double const ct = tables.SampleBremsstrahlungCosTheta(k_eV, u);
                supported = supported && std::isfinite(ct) && ct >= -1.0 && ct <= 1.0;
                monotone = monotone && ct <= previous;
                previous = ct;
                backward += ct == -1.0;
                double const gu = gamma_tot * std::sqrt(2.0 * (1.0 - ct));
                m1 += gu;
                m2 += gu * gu;
            }
            m1 /= n_angles;
            m2 = std::sqrt(m2 / n_angles);
            previous = 1.0;
            for (double u : {0.0, std::nextafter(0.0, 1.0), 0x1.0p-53, 1.0e-12,
                             0.5, 1.0 - 1.0e-12, std::nextafter(1.0, 0.0), 1.0}) {
                double const ct = tables.SampleBremsstrahlungCosTheta(k_eV, u);
                supported = supported && std::isfinite(ct) && ct >= -1.0 && ct <= 1.0;
                monotone = monotone && ct <= previous;
                previous = ct;
            }
            bool const endpoints = tables.SampleBremsstrahlungCosTheta(k_eV, 0.0) == 1.0
                && tables.SampleBremsstrahlungCosTheta(k_eV, 1.0) == -1.0;
            check(supported && monotone && endpoints && backward == 0
                      && std::abs(m1 / tsai_mean - 1.0) <= 0.002
                      && std::abs(m2 / tsai_rms - 1.0) <= 0.002,
                  "bounded_brems_angle_" + fmt(e_mev) + "MeV",
                  "<u>=" + fmt(m1) + " (quadrature " + fmt(tsai_mean) + "), rms="
                      + fmt(m2) + " (quadrature " + fmt(tsai_rms)
                      + "), exactly backward=" + std::to_string(backward));
        }
    }
    exercise_brems_events(tables);

    // ---- In-flight annihilation ---------------------------------------------
    for (double e_mev : {0.3, 1.0, 5.0}) {
        double const k_eV = e_mev * 1.0e6;
        double eps_lo = 0.0;
        double eps_hi = 0.0;
        double mean_ref = 0.0;
        heitler_reference(k_eV, eps_lo, eps_hi, mean_ref);
        double const e_total = k_eV + 2.0 * kMeC2eV;
        int const n1 = 601;
        int const n3 = 11;
        long n_draws = 0;
        long n_out = 0;
        double mean_eps = 0.0;
        double worst_closure = 0.0;
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n3; ++j) {
                double const u1 = (i + 0.5) / n1;
                double const u3 = (j + 0.5) / n3;
                auto const fs = tables.SamplePositronAnnihilationFinalState(k_eV, u1, u3);
                if (!fs.valid) { continue; }
                ++n_draws;
                double const eps = fs.photon1_energy_eV / e_total;
                mean_eps += eps;
                if (eps < eps_lo * (1.0 - 1.0e-9) || eps > eps_hi * (1.0 + 1.0e-9)) { ++n_out; }
                worst_closure = std::max(
                    worst_closure,
                    std::abs((fs.photon1_energy_eV + fs.photon2_energy_eV) / e_total - 1.0));
            }
        }
        mean_eps /= std::max<long>(n_draws, 1);
        check(worst_closure <= 5.0e-9, "annihilation_energy_closure_" + fmt(e_mev) + "MeV",
              "max |sum(E_gamma)/E_total - 1| = " + fmt(worst_closure));
        check(n_out == 0, "annihilation_support_" + fmt(e_mev) + "MeV",
              fmt(100.0 * n_out / std::max<long>(n_draws, 1))
                  + "% of draws outside the kinematic range ["
                  + fmt(eps_lo) + ", " + fmt(eps_hi) + "]");
        check(std::abs(mean_eps / mean_ref - 1.0) <= 0.01,
              "annihilation_mean_fraction_" + fmt(e_mev) + "MeV",
              "sampled <eps> = " + fmt(mean_eps) + " vs Heitler " + fmt(mean_ref)
                  + " (tol 1%)");
    }

    // ---- Annihilation at rest --------------------------------------------
    // The in-flight sweep above starts at 300 keV, so the at-rest branch had
    // never run.  At rest is the physically common case -- it is what a thermalised positron
    // does, and it is the source of the 511 keV feedback line.
    {
        auto const rest = tables.SamplePositronAnnihilationFinalState(0.0, 0.5, 0.25);
        std::ostringstream d;
        d.precision(12);
        d << "photon1 = " << rest.photon1_energy_eV << " eV, photon2 = "
          << rest.photon2_energy_eV << " eV, at_rest = " << rest.at_rest;
        // Relative, not absolute: this smoke carries mec^2 = 510998.95069 while
        // the tables translation unit uses 510998.95000, a 1.4e-9 relative step.
        check(rest.valid && rest.at_rest
                  && std::abs(rest.photon1_energy_eV - kMeC2eV) <= 1.0e-8 * kMeC2eV
                  && std::abs(rest.photon2_energy_eV - kMeC2eV) <= 1.0e-8 * kMeC2eV,
              "annihilation at rest emits two 511 keV photons", d.str());

        // At rest there is no preferred axis, so the polar draw must map
        // straight onto an isotropic cosine.
        int const n = 20001;
        double mean_cos = 0.0, lowest = 2.0, highest = -2.0;
        for (int i = 0; i < n; ++i) {
            auto const st = tables.SamplePositronAnnihilationFinalState(
                0.0, (i + 0.5) / n, 0.25);
            mean_cos += st.photon1_cos_theta;
            lowest = std::min(lowest, st.photon1_cos_theta);
            highest = std::max(highest, st.photon1_cos_theta);
        }
        mean_cos /= n;
        std::ostringstream di;
        di.precision(6);
        di << "<cos> = " << mean_cos << " over [" << lowest << ", " << highest << "]";
        check(std::abs(mean_cos) <= 1.0e-3 && lowest < -0.999 && highest > 0.999,
              "annihilation at rest is isotropic", di.str());
    }

    // ---- Photoelectric subshell and binding energy ---------------------------
    // SamplePhotoelectricFinalState was untested: only the photoelectron ANGLE
    // had coverage.  The subshell draw picks the element and shell, and its
    // binding energy is subtracted from the photon, so a wrong inversion moves
    // the photoelectron spectrum without ever producing an invalid state.
    // Geant4's own final-state events (validation/photon_final_state_geant4_
    // events.csv) show the K-shell binding energies air actually produces:
    // N 404.85 eV, O 537.28 eV, Ar 3177.60 eV.  Those must appear among the
    // sampled shells.
    {
        int const n = 20001;
        for (double e_eV : {1.0e4, 3.0e4, 1.0e5}) {
            std::set<int> elements;
            std::set<std::string> shells;
            double worst_closure = 0.0;
            bool binding_in_range = true;
            std::vector<double> sampled_bindings;
            for (int i = 0; i < n; ++i) {
                auto const st = tables.SamplePhotoelectricFinalState(e_eV, (i + 0.5) / n);
                elements.insert(st.element_Z);
                shells.insert(st.shell_id);
                sampled_bindings.push_back(st.binding_energy_eV);
                if (st.binding_energy_eV <= 0.0 || st.binding_energy_eV >= e_eV) {
                    binding_in_range = false;
                }
                worst_closure = std::max(worst_closure,
                    std::abs(st.photoelectron_energy_eV + st.binding_energy_eV - e_eV) / e_eV);
            }
            std::ostringstream d;
            d.precision(6);
            d << "E=" << e_eV << "eV: " << elements.size() << " elements, " << shells.size()
              << " shells, worst |T + B - E|/E = " << worst_closure;
            check(binding_in_range && worst_closure <= 1.0e-12 && elements.size() >= 2
                      && shells.size() >= 2,
                  "photoelectric final state closes and samples several shells", d.str());
            for (int z : elements) {
                check(z == 6 || z == 7 || z == 8 || z == 18,
                      "photoelectric target is an air element",
                      "E=" + fmt(e_eV) + "eV sampled Z=" + fmt(z));
            }
            // The K edges Geant4 produces in air must be reachable.
            for (double edge : {404.85, 537.28, 3177.60}) {
                if (edge >= e_eV) { continue; }
                bool found = false;
                for (double binding : sampled_bindings) {
                    if (std::abs(binding - edge) <= 0.01 * edge) { found = true; break; }
                }
                check(found, "photoelectric samples the Geant4 K-shell binding",
                      "E=" + fmt(e_eV) + "eV edge " + fmt(edge) + " eV");
            }
        }
    }

    // ---- Photon interaction channel selection --------------------------------
    // A mis-normalised cumulative share shifts the Compton / pair /
    // photoelectric mix and photon-feedback gain. The shares come from
    // PhotonProcessInverseLengths, which is public, so the inversion can be
    // checked directly rather than inferred: each boundary must fall exactly at
    // the cumulative rate fraction, and a fine u-sweep must reproduce the
    // fractions themselves.
    {
        for (double e_mev : {0.5, 5.0, 50.0}) {
            double const e_eV = e_mev * 1.0e6;
            auto const rates = tables.PhotonProcessInverseLengths(e_eV, 1.0);
            double const total = rates.TotalPerM();
            struct ChannelShare {
                rrea::RreaPhotonChannel channel;
                char const* name;
                double rate;
            };
            std::vector<ChannelShare> const shares = {
                {rrea::RreaPhotonChannel::Compton, "compton", rates.compton_per_m},
                {rrea::RreaPhotonChannel::Photoelectric, "photoelectric", rates.photoelectric_per_m},
                {rrea::RreaPhotonChannel::PairNuclear, "pair_nuclear", rates.pair_nuclear_per_m},
                {rrea::RreaPhotonChannel::PairTriplet, "pair_triplet", rates.pair_triplet_per_m},
            };

            // Boundaries: just inside each populated channel's interval the
            // sampler must return that channel.
            double cumulative = 0.0;
            for (auto const& share : shares) {
                double const fraction = share.rate / total;
                if (fraction > 1.0e-9) {
                    double const inside = (cumulative + 0.5 * fraction);
                    auto const got = tables.SamplePhotonChannel(e_eV, inside);
                    std::ostringstream d;
                    d.precision(6);
                    d << "E=" << e_mev << "MeV " << share.name << " share=" << fraction
                      << " u=" << inside << " selected=" << static_cast<int>(got)
                      << " expected=" << static_cast<int>(share.channel);
                    check(got == share.channel,
                          "photon channel boundary matches its rate share", d.str());
                }
                cumulative += fraction;
            }

            // Fine sweep: sampled frequencies must reproduce the rate shares.
            int const n = 200001;
            std::map<int, long> counts;
            for (int i = 0; i < n; ++i) {
                counts[static_cast<int>(
                    tables.SamplePhotonChannel(e_eV, (i + 0.5) / n))] += 1;
            }
            double worst = 0.0;
            char const* worst_name = "";
            for (auto const& share : shares) {
                double const want = share.rate / total;
                double const got =
                    static_cast<double>(counts[static_cast<int>(share.channel)]) / n;
                if (std::abs(got - want) > worst) {
                    worst = std::abs(got - want);
                    worst_name = share.name;
                }
            }
            std::ostringstream d;
            d.precision(6);
            d << "E=" << e_mev << "MeV worst |sampled - rate| = " << worst
              << " on " << worst_name;
            check(worst <= 2.0 / n,
                  "photon channel frequencies reproduce the rate shares", d.str());
        }
    }

    // ---- Compton Klein-Nishina final state (Butcher-Messel rejection) -------
    for (double e_mev : {0.5, 5.0, 50.0}) {
        double const e_eV = e_mev * 1.0e6;
        double const alpha = e_eV / kMeC2eV;
        double mean_eps_ref = 0.0, mean_mu_ref = 0.0, eps_min = 0.0;
        kn_moments(alpha, mean_eps_ref, mean_mu_ref, eps_min);
        double mean_eps = 0.0, mean_mu = 0.0, worst_cons = 0.0;
        long n_out = 0;
        int const n = 200001;
        for (int i = 0; i < n; ++i) {
            double const u1 = (i + 0.5) / n;
            double const u2 = std::fmod(u1 * 7919.0 + 0.5, 1.0);  // decorrelated accept draw
            auto const st = tables.SampleComptonKleinNishinaFinalState(e_eV, u1, u2);
            double const eps = st.scattered_photon_energy_eV / e_eV;
            mean_eps += eps;
            mean_mu += st.cos_theta;
            worst_cons = std::max(worst_cons,
                std::abs((st.scattered_photon_energy_eV + st.recoil_electron_energy_eV) / e_eV - 1.0));
            if (eps < eps_min * (1.0 - 1.0e-6) || eps > 1.0 + 1.0e-6) { ++n_out; }
        }
        mean_eps /= n;
        mean_mu /= n;
        check(worst_cons <= 1.0e-9, "compton_energy_closure_" + fmt(e_mev) + "MeV",
              "max |(E'+T_e)/E - 1| = " + fmt(worst_cons));
        check(n_out == 0, "compton_support_" + fmt(e_mev) + "MeV",
              fmt(100.0 * n_out / n) + "% outside eps in [" + fmt(eps_min) + ", 1]");
        check(std::abs(mean_eps / mean_eps_ref - 1.0) <= 0.02,
              "compton_mean_scattered_fraction_" + fmt(e_mev) + "MeV",
              "sampled <E'/E> = " + fmt(mean_eps) + " vs Klein-Nishina " + fmt(mean_eps_ref)
                  + " (tol 2%)");
        check(std::abs(mean_mu - mean_mu_ref) <= 0.02,
              "compton_mean_costheta_" + fmt(e_mev) + "MeV",
              "sampled <cos theta> = " + fmt(mean_mu) + " vs Klein-Nishina " + fmt(mean_mu_ref)
                  + " (tol 0.02)");
    }

    // ---- Photoelectric Sauter-Gavrila photoelectron angle -------------------
    {
        std::vector<double> const quantiles{0.01, 0.1, 0.5, 0.9, 0.99};
        double prev_mean_mu = -2.0;
        for (double const t_eV : {
                 1.0e3,
                 1.0e5,
                 5.0e5,
                 1.0e6,
                 3.0e6,
                 1.0e7,
                 1.0e9,
                 1.0e10}) {
            std::string const tag = fmt(t_eV) + "eV";
            SauterGavrilaReference const reference =
                sauter_gavrila_reference(t_eV, quantiles);
            double mean_mu = 0.0;
            double mean_mu2 = 0.0;
            long n_out = 0;
            int const n = 20001;
            for (int i = 0; i < n; ++i) {
                double const u = (i + 0.5) / n;
                double const mu = tables.SamplePhotoelectronCosTheta(t_eV, u);
                mean_mu += mu;
                mean_mu2 += mu * mu;
                if (!std::isfinite(mu) || mu < -1.0 || mu > 1.0) { ++n_out; }
            }
            mean_mu /= n;
            mean_mu2 /= n;
            check(n_out == 0, "photoelectron_support_" + tag,
                  fmt(n_out) + " out-of-range cos theta");
            check(std::abs(mean_mu - reference.mean_mu) <= 5.0e-5,
                  "photoelectron_mean_costheta_" + tag,
                  "sampled <cos theta> = " + fmt(mean_mu)
                      + " vs independent log-t quadrature " + fmt(reference.mean_mu)
                      + " (tol 5e-5)");
            check(std::abs(mean_mu2 - reference.mean_mu2) <= 5.0e-5,
                  "photoelectron_second_moment_" + tag,
                  "sampled <cos^2 theta> = " + fmt(mean_mu2)
                      + " vs independent log-t quadrature " + fmt(reference.mean_mu2)
                      + " (tol 5e-5)");
            double max_quantile_error = 0.0;
            for (std::size_t iq = 0; iq < quantiles.size(); ++iq) {
                double const sampled =
                    tables.SamplePhotoelectronCosTheta(t_eV, quantiles[iq]);
                max_quantile_error = std::max(
                    max_quantile_error,
                    std::abs(sampled - reference.quantile_mu[iq]));
            }
            check(max_quantile_error <= 2.0e-5,
                  "photoelectron_quantiles_" + tag,
                  "max |sampled-reference cos theta| = " + fmt(max_quantile_error)
                      + " at u={0.01,0.1,0.5,0.9,0.99} (tol 2e-5)");
            check(
                  tables.SamplePhotoelectronCosTheta(t_eV, 0.0) == -1.0
                      && tables.SamplePhotoelectronCosTheta(t_eV, 1.0) == 1.0,
                  "photoelectron_cdf_endpoints_" + tag,
                  "q(0)=-1 and q(1)=+1 exactly");
            check(mean_mu > prev_mean_mu, "photoelectron_forward_boost_" + tag,
                  "<cos theta> = " + fmt(mean_mu) + " increases with energy (was " + fmt(prev_mean_mu)
                      + ")");
            prev_mean_mu = mean_mu;
        }

        double const low_beta_lo = tables.SamplePhotoelectronCosTheta(0.0, 0.0);
        double const low_beta_mid = tables.SamplePhotoelectronCosTheta(0.0, 0.5);
        double const low_beta_hi = tables.SamplePhotoelectronCosTheta(0.0, 1.0);
        check(
            low_beta_lo == -1.0 && std::abs(low_beta_mid) <= 1.0e-15
                && low_beta_hi == 1.0,
            "photoelectron_low_beta_limit",
            "inverse of F(mu)=(2+3mu-mu^3)/4 gives (-1,0,+1) at u=(0,0.5,1)");
    }

    // ---- Pair production energy sharing --------------------------------------
    // The split empirical channels have distinct contracts: nuclear keeps the
    // symmetric-share physics check, while triplet's spectator makes the split
    // asymmetric so only validity is contractual there.  (Support and closure
    // are producer-guaranteed and deliberately not restated.)
    struct PairCase {
        rrea::RreaPhotonChannel channel;
        char const* label;
        std::vector<double> energies_MeV;
        bool check_symmetry;
    };
    std::vector<PairCase> const pair_cases{
        {rrea::RreaPhotonChannel::PairNuclear, "nuclear_",
         {2.0, 10.0, 100.0}, true},
        {rrea::RreaPhotonChannel::PairTriplet, "triplet_",
         {3.0, 10.0, 100.0}, false},
    };
    for (auto const& pair_case : pair_cases) {
        for (double e_mev : pair_case.energies_MeV) {
        double const e_eV = e_mev * 1.0e6;
        double mean_x = 0.0;
        long n_valid = 0;
        int const n = 100001;
        for (int i = 0; i < n; ++i) {
            double const u = (i + 0.5) / n;
            auto const st = tables.SamplePairFinalState(
                e_eV, pair_case.channel, u, 0.5, 0.5, 0.5, 0.5);
            if (!st.valid) { continue; }
            ++n_valid;
            double const e_e_tot = st.electron_energy_eV + kMeC2eV;
            double const e_p_tot = st.positron_energy_eV + kMeC2eV;
            mean_x += e_e_tot / (e_e_tot + e_p_tot);
        }
        mean_x /= std::max<long>(n_valid, 1);
        std::string const tag = std::string(pair_case.label) + fmt(e_mev) + "MeV";
        check(n_valid == n, "pair_valid_" + tag,
              fmt(n_valid) + " / " + fmt(n) + " valid final states");
        if (pair_case.check_symmetry) {
            check(std::abs(mean_x - 0.5) <= 0.02, "pair_share_symmetry_" + tag,
                  "sampled <x_e-> = " + fmt(mean_x)
                      + " (Geant4 pair sharing ~symmetric, tol 0.02)");
        }
        }
    }

    // ---- Pair production angular final state ---------------------------------
    // Nothing read a direction component before this: every pair test pinned the
    // four angular draws at 0.5 and looked only at energies.  Pair opening angles
    // decide where the secondary e+/e- are injected relative to E, and therefore
    // whether they run away at all, so an unchecked angular final state feeds
    // straight into the avalanche.
    //
    // SamplePairFinalState replays stored Geant4 records rather than sampling a
    // distribution, so comparing it to the bank it replays would only exercise
    // the loader.  The assertions below are the ones that survive that: vectors
    // must be unit-norm, the record draw must actually cover the bank instead of
    // collapsing onto one entry, and the replayed angles must obey the
    // Bethe-Heitler scaling <gamma*theta> ~ 1 with the forward peaking that
    // hardens with energy.
    for (auto const& pair_case : pair_cases) {
        double previous_cos = -2.0;
        for (double e_mev : {3.0, 30.0, 100.0}) {
            double const e_eV = e_mev * 1.0e6;
            int const n = 20001;
            double worst_norm = 0.0;
            double sum_gtheta_e = 0.0, sum_gtheta_p = 0.0, sum_cos_e = 0.0;
            long n_valid = 0;
            std::set<long long> distinct_directions;
            for (int i = 0; i < n; ++i) {
                double const u = (i + 0.5) / n;
                // draw_invariant is what indexes the record bank;
                // draw_lepton_polar/azimuth are discarded by the sampler.
                auto const st = tables.SamplePairFinalState(
                    e_eV, pair_case.channel, u, 0.5, 0.5, 0.5, 0.5);
                if (!st.valid) { continue; }
                ++n_valid;
                auto norm = [&](double x, double y, double z) {
                    return std::sqrt(x * x + y * y + z * z);
                };
                for (double magnitude : {norm(st.electron_dir_x, st.electron_dir_y, st.electron_dir_z),
                                         norm(st.positron_dir_x, st.positron_dir_y, st.positron_dir_z),
                                         norm(st.recoil_dir_x, st.recoil_dir_y, st.recoil_dir_z)}) {
                    worst_norm = std::max(worst_norm, std::abs(magnitude - 1.0));
                }
                double const gamma_e = 1.0 + st.electron_energy_eV / kMeC2eV;
                double const gamma_p = 1.0 + st.positron_energy_eV / kMeC2eV;
                sum_gtheta_e += gamma_e * std::acos(std::min(1.0, std::max(-1.0, st.electron_dir_z)));
                sum_gtheta_p += gamma_p * std::acos(std::min(1.0, std::max(-1.0, st.positron_dir_z)));
                sum_cos_e += st.electron_dir_z;
                distinct_directions.insert(
                    static_cast<long long>(st.electron_dir_z * 1.0e12));
            }
            std::string const tag = std::string(pair_case.label) + fmt(e_mev) + "MeV";
            check(worst_norm <= 1.0e-12, "pair_direction_unit_norm_" + tag,
                  "max |‖d‖ - 1| over e-, e+, recoil = " + fmt(worst_norm));
            // A packing or indexing error that pinned the record index would
            // still return valid, unit-norm directions -- just always the same
            // ones.  Require the sweep to reach a large part of the bank.
            check(static_cast<long>(distinct_directions.size()) > 500,
                  "pair_direction_bank_coverage_" + tag,
                  fmt(static_cast<double>(distinct_directions.size()))
                      + " distinct electron directions from " + fmt(n_valid) + " draws");
            double const mean_gtheta_e = sum_gtheta_e / std::max<long>(n_valid, 1);
            double const mean_gtheta_p = sum_gtheta_p / std::max<long>(n_valid, 1);
            double const mean_cos_e = sum_cos_e / std::max<long>(n_valid, 1);
            check(mean_gtheta_e > 0.2 && mean_gtheta_e < 5.0
                      && mean_gtheta_p > 0.2 && mean_gtheta_p < 5.0,
                  "pair_gamma_theta_scaling_" + tag,
                  "<gamma*theta> e- = " + fmt(mean_gtheta_e)
                      + ", e+ = " + fmt(mean_gtheta_p)
                      + " (Bethe-Heitler theta ~ mc^2/E gives O(1))");
            if (pair_case.check_symmetry) {
                double const asymmetry = std::abs(mean_gtheta_e - mean_gtheta_p)
                    / std::max(mean_gtheta_e, mean_gtheta_p);
                check(asymmetry <= 0.20, "pair_angle_charge_symmetry_" + tag,
                      "|<g.th>e- - <g.th>e+| / max = " + fmt(asymmetry));
            }
            check(mean_cos_e > previous_cos, "pair_forward_peaking_" + tag,
                  "<cos theta_e-> = " + fmt(mean_cos_e)
                      + " (previous energy " + fmt(previous_cos) + ")");
            previous_cos = mean_cos_e;
        }
    }

    // ---- Deterministic mapping pins -----------------------------------------
    {
        double const c1 = tables.SampleComptonKleinNishinaFinalState(5.0e6, 0.3, 0.7).cos_theta;
        double const c2 = tables.SampleComptonKleinNishinaFinalState(5.0e6, 0.3, 0.7).cos_theta;
        check(c1 == c2, "compton_determinism", "reproducible cos theta = " + fmt(c1));
        double const p1 = tables.SamplePhotoelectronCosTheta(1.0e5, 0.3);
        double const p2 = tables.SamplePhotoelectronCosTheta(1.0e5, 0.3);
        check(p1 == p2, "photoelectron_determinism", "reproducible cos theta = " + fmt(p1));
        struct Golden { double got, want; const char* label; };
        const Golden goldens[] = {
            {tables.SampleComptonKleinNishinaFinalState(5.0e6, 0.3, 0.7).cos_theta, GOLD_C1, "compton_golden_5MeV"},
            {tables.SampleComptonKleinNishinaFinalState(50.0e6, 0.9, 0.1).cos_theta, GOLD_C2, "compton_golden_50MeV"},
            {tables.SamplePhotoelectronCosTheta(1.0e5, 0.3), GOLD_P1, "photoelectron_golden_100keV"},
            {tables.SamplePhotoelectronCosTheta(3.0e4, 0.8), GOLD_P2, "photoelectron_golden_30keV"},
        };
        auto fmt9 = [](double v) {
            std::ostringstream ss;
            ss.precision(12);
            ss << std::fixed << v;
            return ss.str();
        };
        for (auto const& g : goldens) {
            check(std::abs(g.got - g.want) <= 1.0e-9, g.label,
                  "got " + fmt9(g.got) + " want " + fmt9(g.want));
        }
    }

    // ---- elastic model branch switch at 100 MeV ------------------------------
    // SampleGsCosTheta selects Goudsmit-Saunderson at or below 100 MeV and
    // WentzelVI above it. An anchor pair across one ulp pins that selector.
    {
        double const transition_eV = 1.0e8;
        double const above_eV = std::nextafter(transition_eV, 1.0e9);
        for (bool is_positron : {false, true}) {
            char const* species = is_positron ? "positron" : "electron";
            double const at = tables.SampleGsCosTheta(is_positron, transition_eV, 10.0, 0.5);
            double const above = tables.SampleGsCosTheta(is_positron, above_eV, 10.0, 0.5);
            std::ostringstream d;
            d.precision(6);
            d << species << " 1-cos at 1e8 eV = " << (1.0 - at)
              << ", one ulp above = " << (1.0 - above);
            check(at != above, "elastic model switches to WentzelVI above 100 MeV", d.str());
        }
    }

    // ---- single elastic scatter: screened-Rutherford shape -----------------
    // The GS check below needs <cos> of a single scatter to build its expected
    // chain.  Averaging sampled quantiles cannot supply it: the quantile
    // function cos(u) = 1 - 2*eta*u/(1 + eta - u) has slope ~2/eta at u -> 1,
    // which is 5e6 at 7.2 MeV, so a uniform u-grid misses the wide-angle tail
    // and underestimates the mean deflection by ~22 percent there -- precisely
    // at the energy that carries RREA.  Since the draw is an exact analytic
    // inverse, recover eta from the median and use the closed forms instead:
    // every other quantile is then predicted exactly (a real shape pin, no
    // external fixture), and <1-cos> = 2*eta*((1+eta)*ln(1+1/eta) - 1) is exact
    // rather than quadrature-limited.
    auto screening_eta = [&tables](bool is_positron, double k_eV) {
        double const median = tables.SampleSingleElasticCosTheta(is_positron, k_eV, 0.5);
        return 0.5 * (1.0 - median) / median;
    };
    {
        for (bool is_positron : {false, true}) {
            char const* species = is_positron ? "positron" : "electron";
            double prev_defl = 0.0;
            for (double k_eV : {1.0e5, 1.0e6, 7.2e6, 1.5e8, 1.0e9, 5.0e9}) {
                double const e_mev = k_eV * 1.0e-6;
                double const eta = screening_eta(is_positron, k_eV);
                {
                    // The median-recovered eta pins the quantile SHAPE only; a
                    // wrong screening-parameter magnitude would satisfy every
                    // shape check.  Anchor it to the table's own accessor so
                    // the two cannot silently diverge.
                    double const eta_table =
                        tables.ScreeningEta(is_positron, k_eV);
                    std::ostringstream de;
                    de.precision(12);
                    de << std::scientific << species << " E=" << e_mev
                       << "MeV eta(median)=" << eta
                       << " ScreeningEta=" << eta_table;
                    // 1e-3 relative: the median recovery cancels ~eps/(2 eta)
                    // of precision at GeV-scale eta (~1e-11), while a real
                    // magnitude bug in the screening table is >= percent-level.
                    check(std::abs(eta - eta_table)
                              <= 1.0e-3 * std::max(eta_table, 1.0e-30),
                          "median-recovered eta matches the screening table",
                          de.str());
                }
                for (double u : {0.1, 0.9, 0.99}) {
                    double const got = tables.SampleSingleElasticCosTheta(is_positron, k_eV, u);
                    double const want = 1.0 - 2.0 * eta * u / (1.0 + eta - u);
                    std::ostringstream d;
                    d.precision(12);
                    d << std::scientific << species << " E=" << e_mev << "MeV u=" << u
                      << " cos=" << got << " screened-Rutherford=" << want;
                    check(std::abs(got - want) <= 1.0e-12 * std::max(1.0, std::abs(want)),
                          "single elastic follows the screened-Rutherford quantile", d.str());
                }
                double const defl = 2.0 * eta * ((1.0 + eta) * std::log1p(1.0 / eta) - 1.0);
                if (prev_defl > 0.0) {
                    std::ostringstream dp;
                    dp.precision(4);
                    dp << std::scientific << species << " E=" << e_mev
                       << "MeV eta=" << eta << " mean(1-cos)=" << defl
                       << " previous=" << prev_defl;
                    check(defl > 0.0 && defl < prev_defl,
                          "single elastic deflection falls with rising energy", dp.str());
                }
                prev_defl = defl;
            }
        }
    }

    // ---- GS multiple-scattering path scaling -------------------------------
    // Timestep independence of the transport depends on this.  The angle CDF
    // stores hard bounds at u = 0 (one_minus_cos = 2, full backscatter) and
    // u = 1 (1e-15) in every one of its (E, n) groups.  Sampling that treated
    // the u = 0 bound as data smeared a backward hemisphere over the whole
    // first bin, and since that bound is 2.0 at every path length the injected
    // deflection did not scale with n -- 99 percent of the mean deflection at
    // 7.5 MeV, identical for a long chord and a short one.  Splitting a chord
    // into k pieces then multiplied the scattering by ~k, so halving dt roughly
    // doubled the elastic deflection and collapsed avalanche multiplication.
    //
    // Physical requirement, independent of the interpolation used: the mean
    // deflection must grow with path and match a compound-Poisson chain of
    // single scatters, <cos> = exp(-n (1 - <cos_single>)).
    //
    // Both species are swept: positrons carry their own GS and single-elastic
    // tables, so an electron-only sweep leaves the identical inversion
    // unexercised on half the data it runs against in production.
    {
        int const gs_samples = 20001;
        for (bool is_positron : {false, true}) {
            char const* species = is_positron ? "positron" : "electron";
            for (double k_eV : {1.0e5, 1.0e6, 7.2e6, 1.5e8, 1.0e9, 5.0e9}) {
                double const e_mev = k_eV * 1.0e-6;
                double const eta = screening_eta(is_positron, k_eV);
                double const mu1 =
                    1.0 - 2.0 * eta * ((1.0 + eta) * std::log1p(1.0 / eta) - 1.0);
                double prev_n = 0.0;
                double prev_defl = 0.0;
                for (double n : {10.0, 40.0, 160.0}) {
                    double gs = 0.0;
                    for (int i = 0; i < gs_samples; ++i) {
                        double const u = (i + 0.5) / gs_samples;
                        gs += tables.SampleGsCosTheta(is_positron, k_eV, n, u);
                    }
                    gs /= static_cast<double>(gs_samples);
                    double const defl = 1.0 - gs;
                    // Exact Legendre composition law for n independent
                    // scatters: <cos theta_n> = <cos theta_1>^n.
                    double const expect = 1.0 - std::pow(mu1, n);
                    std::ostringstream d1;
                    d1.precision(4);
                    d1 << std::scientific << species << " E=" << e_mev
                       << "MeV n=" << n << " mean(1-cos)=" << defl
                       << " chain=" << expect
                       << " ratio=" << (expect > 0.0 ? defl / expect : 0.0);
                    // The exact chain leaves table interpolation as the residual.
                    check(expect > 0.0 && defl > 0.65 * expect && defl < 1.5 * expect,
                          "GS mean deflection matches single-scatter chain", d1.str());
                    if (prev_defl > 0.0 && defl > 0.0) {
                        double const slope =
                            std::log(defl / prev_defl) / std::log(n / prev_n);
                        std::ostringstream d2;
                        d2.precision(3);
                        d2 << std::fixed << species << " E=" << e_mev
                           << "MeV n=" << prev_n << "->" << n
                           << " dln(1-cos)/dln(n)=" << slope
                           << " (diffusive=1; path-independent artifact=0)";
                        check(slope > 0.7 && slope < 1.3,
                              "GS deflection scales with path length", d2.str());
                    }
                    prev_n = n;
                    prev_defl = defl;
                }
            }
        }
    }

    // ---- Total collision stopping power closes against NIST ESTAR ----
    //
    // The RREA threshold field is set by the MINIMUM of the total collision
    // drag, so if that minimum is wrong the whole avalanche is wrong. schema-6
    // ships only the restricted eIoni column, and Geant4's unrestricted value
    // was computed during table generation and then discarded, so nothing in
    // the bundle pins this. The closure is
    //     restricted(E) + <T>(E)/lambda_hardMoller(E)  ==  unrestricted(E)
    // and NIST ESTAR gives the dry-air minimum as about 1.66 MeV cm^2/g near
    // 1.2-1.5 MeV. At the configuration's reference density that is
    //     1.66 MeV cm^2/g * 1.20479e-3 g/cm^3 * 1e2 cm/m = 200.0 kV/m.
    // 2 percent is the honest bound: ESTAR's own tabulation is quoted to three
    // figures and the minimum is shallow.
    {
        double const reference_density_g_cm3 = 1.20479e-3;
        double best_kv_per_m = 1.0e300;
        double best_mev_cm2_g = 0.0;
        double best_energy_mev = 0.0;
        double best_restricted_kv_per_m = 0.0;
        double best_discrete_kv_per_m = 0.0;
        for (int i = 0; i <= 400; ++i) {
            // 0.25 MeV .. 10 MeV: the minimum is interior to this window, so a
            // bracketing failure shows up as an endpoint hit rather than a pass.
            double const e_eV = 2.5e5 * std::pow(40.0, i / 400.0);
            double const restricted =
                tables.ElectronCollisionLoss(e_eV, 1.0);
            double const discrete =
                tables.ElectronHardMollerEnergyLoss(e_eV, 1.0);
            double const total = restricted + discrete;
            if (total < best_kv_per_m) {
                best_kv_per_m = total;
                best_energy_mev = e_eV * 1.0e-6;
                best_restricted_kv_per_m = restricted;
                best_discrete_kv_per_m = discrete;
            }
        }
        best_mev_cm2_g = best_kv_per_m / (reference_density_g_cm3 * 1.0e2 * 1.0e6);
        std::ostringstream d;
        d.precision(4);
        d << std::fixed << "min total collision drag = "
          << best_kv_per_m * 1.0e-3 << " kV/m at " << best_energy_mev
          << " MeV (restricted " << best_restricted_kv_per_m * 1.0e-3
          << " + discrete delta rays " << best_discrete_kv_per_m * 1.0e-3
          << ") = " << best_mev_cm2_g << " MeV cm^2/g vs ESTAR 1.66";
        check(std::abs(best_mev_cm2_g - 1.66) <= 0.02 * 1.66,
              "total collision stopping power matches NIST ESTAR dry air", d.str());
        check(best_energy_mev > 1.0 && best_energy_mev < 2.0,
              "collision stopping minimum sits where ESTAR puts it", d.str());
        // The discrete term is independently required; a structural zero would
        // understate drag and make the local-runaway gate fire too easily.
        check(best_discrete_kv_per_m > 0.05 * best_restricted_kv_per_m,
              "discrete delta-ray drag is present, not a structural zero", d.str());
    }

    // ---- Production-bundle quarter-point replay: every charged rate and drag ----
    //
    // The bundle ships a Geant4-referenced holdout
    // (validation/charged_transport_quarterpoints_air.csv.rtb: every charged
    // species/channel). Container validation cannot see
    // the runtime evaluation path (interpolation, density-ratio scaling, unit
    // convention), so replay it
    // against the LOADED accessors so a wrong column, shifted grid, unit slip,
    // or wrong density exponent in any charged channel fails loudly.
    // reference_value is the independent Geant4 side; runtime_value is the
    // generator's own self-check and is deliberately unused.
    {
        std::string const config_path(argv[1]);
        std::size_t const cut = config_path.find_last_of("/\\");
        std::string const validation_csv =
            (cut == std::string::npos ? std::string()
                                      : config_path.substr(0, cut + 1))
            + "validation/charged_transport_quarterpoints_air.csv.rtb";
        auto const holdout = rrea::table_container::Read(validation_csv);
        auto const holdout_energy =
            rrea::table_container::Column(holdout, "energy_eV");
        auto const holdout_reference =
            rrea::table_container::Column(holdout, "reference_value");
        auto const holdout_relerr =
            rrea::table_container::Column(holdout, "relative_error");
        auto const holdout_species =
            rrea::table_container::TextColumn(holdout, "species");
        auto const holdout_channel =
            rrea::table_container::TextColumn(holdout, "channel");
        std::size_t const holdout_rows = rrea::table_container::Rows(holdout);
        check(holdout_energy.size() == holdout_rows
                  && holdout_reference.size() == holdout_rows
                  && holdout_relerr.size() == holdout_rows
                  && holdout_species.size() == holdout_rows
                  && holdout_channel.size() == holdout_rows,
              "quarterpoints holdout columns agree with its row count",
              validation_csv);

        auto evaluate = [&tables](bool positron, std::string const& channel,
                                  double e_eV, double density_ratio) {
            if (channel == "collision_stopping_eV_per_m_stp") {
                return positron
                    ? double(tables.PositronCollisionLoss(e_eV, density_ratio))
                    : double(tables.ElectronCollisionLoss(e_eV, density_ratio));
            }
            if (channel == "soft_radiative_stopping_eV_per_m_stp") {
                return positron
                    ? double(tables.PositronSoftRadiativeDrag(e_eV, density_ratio))
                    : double(tables.ElectronSoftRadiativeDrag(e_eV, density_ratio));
            }
            if (channel == "elastic_transport_inverse_m_stp") {
                return 1.0 / double(positron
                    ? tables.PositronElasticTransportMeanFreePath(e_eV, density_ratio)
                    : tables.ElectronElasticTransportMeanFreePath(e_eV, density_ratio));
            }
            if (channel == "tracked_brems_inverse_m_stp") {
                return 1.0 / double(positron
                    ? tables.PositronBremsstrahlungMeanFreePath(e_eV, density_ratio)
                    : tables.ElectronBremsstrahlungMeanFreePath(e_eV, density_ratio));
            }
            if (channel == "hard_moller_inverse_m_stp") {
                return 1.0 / double(tables.ElectronHardMollerMeanFreePath(e_eV, density_ratio));
            }
            if (channel == "hard_bhabha_inverse_m_stp") {
                return 1.0 / double(tables.PositronHardBhabhaMeanFreePath(e_eV, density_ratio));
            }
            if (channel == "annihilation_inverse_m_stp") {
                return 1.0 / double(tables.PositronAnnihilationMeanFreePath(e_eV, density_ratio));
            }
            return -1.0;  // unknown channel: reported below
        };

        struct ChannelTally {
            long rows = 0;
            long bad = 0;
            double worst_excess = 0.0;  // max over rows of rel_err / tolerance
            double linear_probe_e_eV = 0.0;
            std::string worst;
        };
        std::map<std::string, ChannelTally> tallies;
        long unknown_channels = 0;
        long total_rows = 0;
        for (std::size_t row = 0; row < holdout_rows; ++row) {
            std::string const& species = holdout_species[row];
            std::string const& channel = holdout_channel[row];
            double const e_eV = static_cast<double>(holdout_energy[row]);
            double const reference = static_cast<double>(holdout_reference[row]);
            double const generation_relerr =
                std::abs(static_cast<double>(holdout_relerr[row]));
            std::string const energy_s = std::to_string(e_eV);
            bool const positron = species == "positron";
            double const value = evaluate(positron, channel, e_eV, 1.0);
            auto& tally = tallies[species + "/" + channel];
            ++tally.rows;
            ++total_rows;
            if (value < 0.0) { ++unknown_channels; continue; }
            // The holdout samples the hard-Moller/Bhabha openings femto-eV
            // above threshold, where rates are 1e-21..1e-14 per metre -- mean
            // free paths of >= 1e14 m, unreachable in any km-scale domain and
            // numerically chaotic in relative terms.  Below 1e-9 per metre
            // (an MFP of a million kilometres) both sides only have to agree
            // the channel is effectively closed; every physically reachable
            // rate is held to the strict tolerance.
            constexpr double negligible_inverse_m = 1.0e-9;
            if (reference < negligible_inverse_m) {
                if (!(value < 10.0 * negligible_inverse_m)) {
                    ++tally.bad;
                    if (tally.worst.empty()) {
                        tally.worst = "open channel at closed-channel energy "
                            + energy_s;
                    }
                }
                continue;
            }
            if (tally.linear_probe_e_eV == 0.0 && tally.rows > 3000) {
                tally.linear_probe_e_eV = e_eV;  // a mid-range in-support energy
            }
            // 3x the generation-time residual, floored at 1e-9: the accessor
            // re-runs the same interpolation the generator recorded, so only
            // an evaluation-path or table-content change can exceed this.
            double const tolerance = std::max(3.0 * generation_relerr, 1.0e-9);
            double const relative = std::abs(value / reference - 1.0);
            double const excess = relative / tolerance;
            if (excess > tally.worst_excess) {
                tally.worst_excess = excess;
                std::ostringstream w;
                w.precision(6);
                w << std::scientific << "E=" << e_eV << " eV: |value/ref-1|="
                  << relative << " tol=" << tolerance;
                tally.worst = w.str();
            }
            if (relative > tolerance) { ++tally.bad; }
        }
        check(unknown_channels == 0,
              "quarterpoints holdout contains only known charged channels",
              std::to_string(unknown_channels) + " unknown-channel rows");
        // 11 species-channel curves is a schema property and stays pinned.  The
        // energy count is NOT: the charged grid carries onset anchors at the
        // cuts, so it moves whenever a bundle is regenerated at a different
        // threshold. Derive it and require every curve to sweep the same grid.
        long rows_per_curve = tallies.empty() ? 0 : tallies.begin()->second.rows;
        bool uniform_grid = !tallies.empty() && rows_per_curve > 0;
        for (auto const& [key, tally] : tallies) {
            uniform_grid = uniform_grid && tally.rows == rows_per_curve;
        }
        check(tallies.size() == 11U && uniform_grid
                  && total_rows
                      == static_cast<long>(tallies.size()) * rows_per_curve,
              "quarterpoints holdout is 11 curves over one common grid",
              std::to_string(tallies.size()) + " curves x "
                  + std::to_string(rows_per_curve) + " energies, "
                  + std::to_string(total_rows) + " rows read");
        for (auto const& [key, tally] : tallies) {
            std::ostringstream d;
            d << tally.rows << " rows, " << tally.bad << " outside tolerance, worst "
              << (tally.worst.empty() ? std::string("-") : tally.worst);
            check(tally.bad == 0,
                  "quarterpoints replay: " + key + " matches its Geant4 reference",
                  d.str());
            // The density exponent lives in the accessor's C++ multiply, so
            // rates and stopping powers must both scale exactly linearly.
            if (tally.linear_probe_e_eV > 0.0) {
                std::string const channel = key.substr(key.find('/') + 1);
                double const full =
                    evaluate(key.compare(0, 8, "positron") == 0, channel,
                             tally.linear_probe_e_eV, 1.0);
                double const half =
                    evaluate(key.compare(0, 8, "positron") == 0, channel,
                             tally.linear_probe_e_eV, 0.5);
                std::ostringstream l;
                l.precision(12);
                l << std::scientific << "E=" << tally.linear_probe_e_eV
                  << " eV: value(rho/2)/value(rho)=" << half / full;
                check(std::abs(half / full - 0.5) <= 1.0e-12,
                      "quarterpoints replay: " + key + " scales linearly in density",
                      l.str());
            }
        }
    }

    if (g_failures > 0) {
        std::cout << "rrea_scattering_sampler_smoke: " << g_failures
                  << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "rrea_scattering_sampler_smoke passed\n";
    return EXIT_SUCCESS;
}
