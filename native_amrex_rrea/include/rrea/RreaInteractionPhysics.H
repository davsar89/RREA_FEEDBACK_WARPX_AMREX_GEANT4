#ifndef RREA_INTERACTION_PHYSICS_H
#define RREA_INTERACTION_PHYSICS_H
// Single numerical implementation for host and CUDA: the schema-6 sampler
// equations moved here unchanged from RreaInteractionTables.cpp. File parsing,
// metadata and validation remain host responsibilities. No table extrapolation
// or process substitution is introduced by device execution.
#include "rrea/RreaInteractionTypes.H"
#include "rrea/RreaDebugOptions.H"
#include "rrea/RreaUniformStream.H"
#include <cmath>
#include <limits>
namespace rrea {
struct RreaMollerCdfGroupView {
    amrex::Real primary_energy_eV = 0.0;
    KineticSpan<amrex::Real const> cdf_u;
    KineticSpan<amrex::Real const> secondary_energy_eV;
};

struct RreaConditionalCdfGroupView {
    amrex::Real primary_energy_eV = 0.0;
    KineticSpan<amrex::Real const> cdf_u;
    KineticSpan<amrex::Real const> values;
};

struct RreaElasticMfpTableView {
    KineticSpan<amrex::Real const> energy_eV;
    KineticSpan<amrex::Real const> elastic_mfp_m_stp;
    KineticSpan<amrex::Real const> screening_eta;
    KineticSpan<amrex::Real const> transport_mfp_m_stp;
};

struct RreaGsAngleTableView {
    KineticSpan<amrex::Real const> energy_eV;   // sorted grid
    KineticSpan<amrex::Real const> path_n;      // sorted grid (s/lambda_el)
    KineticSpan<amrex::Real const> cdf_u;       // shared CDF nodes (strictly increasing)
    KineticSpan<amrex::Real const> one_minus_cos;  // flattened (energy, path, u)
};

struct RreaPhotoelectricSubshellCdfRowView {
    amrex::Real cdf_u = 0.0;
    amrex::Real subshell_probability = 0.0;
    int element_Z = 0;
    amrex::Real binding_energy_eV = 0.0;
    amrex::Real relaxation_energy_eV = 0.0;
};

struct RreaPhotoelectricSubshellCdfGroupView {
    amrex::Real primary_energy_eV = 0.0;
    KineticSpan<RreaPhotoelectricSubshellCdfRowView const> rows;
};

struct RreaPairEmpiricalBlockView {
    RreaPhotonChannel channel = RreaPhotonChannel::PairNuclear;
    amrex::Real primary_energy_eV = 0.0;
    KineticSpan<RreaPairEmpiricalRecord const> training_records;
};

struct RreaPhotoelectricNumericState {
    amrex::Real photoelectron_energy_eV=0, binding_energy_eV=0, relaxation_energy_eV=0;
    int element_Z=0;
    std::size_t group_index=0,row_index=0;
};

struct RreaInteractionTableView {
    RreaInterpolationMode mode=RreaInterpolationMode::LogLogLinear;
    KineticSpan<amrex::Real const> energy_eV,values,log_energy,log_values;
    RREA_HOST_DEVICE amrex::Real Interpolate(amrex::Real energy_eV_query) const;
};
namespace kinetic_tables {
inline constexpr amrex::Real kElectronRestEnergyEv=amrex::Real(rrea::me_c2_eV);
inline constexpr amrex::Real kPairNuclearThresholdEv=2*kElectronRestEnergyEv;
inline constexpr amrex::Real kPairTripletThresholdEv=4*kElectronRestEnergyEv;
inline constexpr amrex::Real kTwoPi=amrex::Real(6.283185307179586476925286766559);
RREA_HOST_DEVICE inline amrex::Real clamp_unit(amrex::Real value)
{
    if (!std::isfinite(static_cast<double>(value))) {
        return amrex::Real(0.5);
    }
    return rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), value));
}

RREA_HOST_DEVICE inline amrex::Real wrap_unit_interval(amrex::Real value)
{
    if (!std::isfinite(static_cast<double>(value))) {
        return amrex::Real(0.0);
    }
    value = value - std::floor(value);
    if (value < amrex::Real(0.0)) {
        value += amrex::Real(1.0);
    }
    return value;
}

RREA_HOST_DEVICE inline amrex::Real sample_annihilation_photon_energy_fraction(
    amrex::Real gamma,
    amrex::Real draw_energy)
{
    // Exact Heitler spectrum. The support uses the CM beta; using the lab beta
    // admits kinematically impossible draws. The nonconstant rejection kernel
    // is required in addition to the 1/eps proposal:
    //   f(eps) ~ (1/eps) * g(eps),  g(eps) = 1 + 2g/(g+1)^2 - eps
    //                                        - 1/((g+1)^2 eps)
    // over the physical support eps in [.5(1-b_cm), .5(1+b_cm)],
    // b_cm = sqrt((g-1)/(g+1)), by deterministic bisection on the analytic
    // antiderivative F(eps) = a ln eps - eps + 1/((g+1)^2 eps),
    // a = 1 + 2g/(g+1)^2.  The inversion consumes one uniform, no rejection.
    if (gamma <= amrex::Real(1.0 + 1.0e-12)) {
        return amrex::Real(0.5);
    }
    amrex::Real const beta_cm = std::sqrt(rrea::kinetic::max(
        amrex::Real(0.0),
        (gamma - amrex::Real(1.0)) / (gamma + amrex::Real(1.0))));
    amrex::Real const eps_min = amrex::Real(0.5) * (amrex::Real(1.0) - beta_cm);
    amrex::Real const eps_max = amrex::Real(0.5) * (amrex::Real(1.0) + beta_cm);
    amrex::Real const u = clamp_unit(draw_energy);
    amrex::Real const gp1 = gamma + amrex::Real(1.0);
    amrex::Real const a = amrex::Real(1.0) + amrex::Real(2.0) * gamma / (gp1 * gp1);
    auto antiderivative = [a, gp1](amrex::Real e) {
        return a * std::log(e) - e + amrex::Real(1.0) / (gp1 * gp1 * e);
    };
    amrex::Real const f_lo = antiderivative(eps_min);
    amrex::Real const f_hi = antiderivative(eps_max);
    amrex::Real const span = f_hi - f_lo;
    if (!(span > amrex::Real(0.0))) {
        return amrex::Real(0.5);
    }
    amrex::Real const target = f_lo + u * span;
    amrex::Real lo = eps_min;
    amrex::Real hi = eps_max;
    for (int iter = 0; iter < 64; ++iter) {
        amrex::Real const mid = amrex::Real(0.5) * (lo + hi);
        if (antiderivative(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return rrea::kinetic::max(eps_min, rrea::kinetic::min(eps_max, amrex::Real(0.5) * (lo + hi)));
}

RREA_HOST_DEVICE inline amrex::Real mean_free_path_from_inverse_length(amrex::Real inverse_length_per_m)
{
    if (!std::isfinite(static_cast<double>(inverse_length_per_m))
        || inverse_length_per_m < amrex::Real(0.0)) {
        rrea::KineticFail("RREA process inverse length must be finite and nonnegative");
    }
    return inverse_length_per_m > amrex::Real(0.0)
        ? amrex::Real(1.0) / inverse_length_per_m
        : amrex::Real(1.0e99);
}

RREA_HOST_DEVICE inline void normalize_local_direction(amrex::Real& x, amrex::Real& y, amrex::Real& z)
{
    amrex::Real const norm = std::sqrt(x * x + y * y + z * z);
    if (norm <= amrex::Real(0.0) || !std::isfinite(static_cast<double>(norm))) {
        x = amrex::Real(0.0);
        y = amrex::Real(0.0);
        z = amrex::Real(1.0);
        return;
    }
    x /= norm;
    y /= norm;
    z /= norm;
}

RREA_HOST_DEVICE inline amrex::Real interp_semilogx_strict(
    KineticSpan<amrex::Real const> const& x,
    KineticSpan<amrex::Real const> const& y,
    amrex::Real q)
{
    if (x.size() < 2) {
        return y.empty() ? amrex::Real(0.0) : y.front();
    }
    if (!std::isfinite(static_cast<double>(q)) || q < x.front() || q > x.back()) {
        rrea::KineticFail("RREA elastic-table query is outside its inclusive energy range");
    }
    if (q == x.front()) return y.front();
    if (q == x.back()) return y.back();
    auto const upper = rrea::kinetic::lower_bound(x.begin(), x.end(), q);
    std::size_t const hi = static_cast<std::size_t>(rrea::kinetic::distance(x.begin(), upper));
    std::size_t const lo = hi - 1U;
    // Node-exact fast path (every other interpolator here has one).  Without
    // it a query at the nextafter side of a sub-ulp anchor pair (the exact
    // 1e8 GS/Wentzel transition rows in the elastic tables) divides by a
    // zero log-segment width because log(1e8) and log(nextafter(1e8)) are the
    // same double.
    if (x[hi] == q) return y[hi];
    amrex::Real const t = (std::log(q) - std::log(x[lo])) / (std::log(x[hi]) - std::log(x[lo]));
    return y[lo] + t * (y[hi] - y[lo]);
}

// Bracket q in a sorted grid; returns lower index and log-fraction to next node.
RREA_HOST_DEVICE inline void log_bracket(KineticSpan<amrex::Real const> const& g, amrex::Real q, int& i, amrex::Real& frac)
{
    int const last = static_cast<int>(g.size()) - 1;
    if (!std::isfinite(static_cast<double>(q)) || q < g[0] || q > g[last]) {
        rrea::KineticFail("RREA elastic-angle query is outside its inclusive tabulated range");
    }
    if (q == g[0]) { i = 0; frac = amrex::Real(0.0); return; }
    if (q == g[last]) { i = last - 1; frac = amrex::Real(1.0); return; }
    int lo = 0, hi = last;
    while (hi - lo > 1) { int const m = (lo + hi) / 2; if (g[m] <= q) lo = m; else hi = m; }
    i = lo;
    amrex::Real const log_lo = std::log(g[lo]);
    amrex::Real const log_hi = std::log(g[hi]);
    if (log_hi == log_lo) {
        // Sub-ulp bracket (duplicated-edge/nextafter anchor rows): the only
        // representable queries inside it are its endpoints.
        frac = (q == g[lo]) ? amrex::Real(0.0) : amrex::Real(1.0);
        return;
    }
    frac = (std::log(q) - log_lo) / (log_hi - log_lo);
}

// Sample cos(theta) from the 2D GS inverse CDF: bilinear in (log E, log n) of
// log(one_minus_cos) at fixed u-nodes, then linear in cos(theta) across u.
RREA_HOST_DEVICE inline amrex::Real sample_gs_angle_cos(
    RreaGsAngleTableView const& t, amrex::Real energy_eV, amrex::Real path_n, amrex::Real u)
{
    int const np = static_cast<int>(t.path_n.size());
    int const nu = static_cast<int>(t.cdf_u.size());
    int ie = 0, ip = 0;
    amrex::Real te = 0.0, tn = 0.0;
    log_bracket(t.energy_eV, energy_eV, ie, te);
    log_bracket(t.path_n, path_n, ip, tn);
    auto cos_at_node = [&](int iu) -> amrex::Real {
        auto val = [&](int e_i, int p_i) -> amrex::Real {
            return t.one_minus_cos[(static_cast<std::size_t>(e_i) * np + p_i) * nu + iu];
        };
        amrex::Real const la = std::log(val(ie, ip));
        amrex::Real const lb = std::log(val(ie + 1, ip));
        amrex::Real const lc = std::log(val(ie, ip + 1));
        amrex::Real const ld = std::log(val(ie + 1, ip + 1));
        amrex::Real const lx = (1 - te) * (1 - tn) * la + te * (1 - tn) * lb
                             + (1 - te) * tn * lc + te * tn * ld;
        return amrex::Real(1.0) - std::exp(lx);
    };
    // u nodes 0 and nu-1 are hard bounds, not samples: every (E, n) group stores
    // one_minus_cos = 2.0 at u = 0 and 1e-15 at u = 1.  The interior nodes are
    // diffusive (mean deflection scales as n^1.05 to n^1.14) and the resolved
    // tail follows one_minus_cos ~ 1/u, so interpolate log(one_minus_cos)
    // against log(u) over interior nodes only -- matching the log treatment this
    // function already uses on the E and n axes -- and continue that power law
    // below the first interior node.  The 2.0 bound is enforced as a clamp only.
    auto one_minus_cos_at = [&](int iu) -> amrex::Real {
        return rrea::kinetic::max(amrex::Real(1.0) - cos_at_node(iu),
                        std::numeric_limits<amrex::Real>::min());
    };
    int const first = 1;          // first interior node
    int const last = nu - 2;      // last interior node
    amrex::Real one_minus_cos;
    if (last <= first) {
        one_minus_cos = one_minus_cos_at(first);
    } else if (u <= t.cdf_u[first]) {
        // Extreme tail: continue the resolved power law rather than the bound.
        amrex::Real const d1 = one_minus_cos_at(first);
        amrex::Real const d2 = one_minus_cos_at(first + 1);
        amrex::Real const u1 = t.cdf_u[first];
        amrex::Real const u2 = t.cdf_u[first + 1];
        amrex::Real const slope = std::log(d2 / d1) / std::log(u2 / u1);
        amrex::Real const u_eff = rrea::kinetic::max(u, u1 * amrex::Real(1.0e-6));
        one_minus_cos = d1 * std::exp(slope * std::log(u_eff / u1));
    } else if (u >= t.cdf_u[last]) {
        one_minus_cos = one_minus_cos_at(last);
    } else {
        int lo = first, hi = last;
        while (hi - lo > 1) { int const m = (lo + hi) / 2; if (t.cdf_u[m] <= u) lo = m; else hi = m; }
        amrex::Real const d0 = one_minus_cos_at(lo);
        amrex::Real const d1 = one_minus_cos_at(hi);
        amrex::Real const fu =
            std::log(u / t.cdf_u[lo]) / std::log(t.cdf_u[hi] / t.cdf_u[lo]);
        one_minus_cos = d0 * std::exp(fu * std::log(d1 / d0));
    }
    amrex::Real const cos_val = amrex::Real(1.0)
        - rrea::kinetic::min(amrex::Real(2.0), one_minus_cos);
    return rrea::kinetic::min(amrex::Real(1.0), rrea::kinetic::max(amrex::Real(-1.0), cos_val));
}

RREA_HOST_DEVICE inline amrex::Real sample_conditional_cdf(
    KineticSpan<RreaConditionalCdfGroupView const> const& groups,
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u,
    bool log_interp)
{
    if (groups.empty()) {
        rrea::KineticFail("RREA conditional CDF table is not loaded");
    }
    cdf_u = rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), cdf_u));

    auto const sample_group = [cdf_u](RreaConditionalCdfGroupView const& group) {
        auto const upper = rrea::kinetic::lower_bound(group.cdf_u.begin(), group.cdf_u.end(), cdf_u);
        if (upper == group.cdf_u.begin()) {
            return group.values.front();
        }
        if (upper == group.cdf_u.end()) {
            return group.values.back();
        }
        std::size_t const hi =
            static_cast<std::size_t>(rrea::kinetic::distance(group.cdf_u.begin(), upper));
        std::size_t const lo = hi - 1U;
        amrex::Real const denom = group.cdf_u[hi] - group.cdf_u[lo];
        amrex::Real const t = denom > amrex::Real(0.0)
            ? (cdf_u - group.cdf_u[lo]) / denom
            : amrex::Real(0.0);
        return group.values[lo] + t * (group.values[hi] - group.values[lo]);
    };

    if (!std::isfinite(static_cast<double>(primary_energy_eV))
        || primary_energy_eV < groups.front().primary_energy_eV
        || primary_energy_eV > groups.back().primary_energy_eV) {
        rrea::KineticFail("RREA conditional-CDF primary energy ");
    }
    if (primary_energy_eV == groups.front().primary_energy_eV) {
        return sample_group(groups.front());
    }
    if (primary_energy_eV == groups.back().primary_energy_eV) {
        return sample_group(groups.back());
    }

    auto const upper = rrea::kinetic::lower_bound(
        groups.begin(),
        groups.end(),
        primary_energy_eV,
        [](RreaConditionalCdfGroupView const& group, amrex::Real energy) {
            return group.primary_energy_eV < energy;
        });
    if (upper == groups.begin()) {
        return sample_group(*upper);
    }
    if (upper == groups.end()) {
        return sample_group(groups.back());
    }
    std::size_t const hi = static_cast<std::size_t>(rrea::kinetic::distance(groups.begin(), upper));
    std::size_t const lo = hi - 1U;
    if (groups[hi].primary_energy_eV == primary_energy_eV) {
        return sample_group(groups[hi]);
    }
    amrex::Real const y0 = sample_group(groups[lo]);
    amrex::Real const y1 = sample_group(groups[hi]);
    amrex::Real const t =
        (std::log(primary_energy_eV) - std::log(groups[lo].primary_energy_eV))
        / (std::log(groups[hi].primary_energy_eV) - std::log(groups[lo].primary_energy_eV));
    if (log_interp && y0 > amrex::Real(0.0) && y1 > amrex::Real(0.0)) {
        return std::exp(std::log(y0) + t * (std::log(y1) - std::log(y0)));
    }
    return y0 + t * (y1 - y0);
}

RREA_HOST_DEVICE inline amrex::Real mean_conditional_cdf(
    KineticSpan<RreaConditionalCdfGroupView const> const& groups,
    amrex::Real primary_energy_eV)
{
    auto group_mean = [](RreaConditionalCdfGroupView const& group) {
        amrex::Real mean = group.cdf_u.front() * group.values.front();
        for (std::size_t i = 1; i < group.cdf_u.size(); ++i) {
            mean += amrex::Real(0.5)
                * (group.values[i - 1] + group.values[i])
                * (group.cdf_u[i] - group.cdf_u[i - 1]);
        }
        mean += (amrex::Real(1.0) - group.cdf_u.back())
            * group.values.back();
        return mean;
    };
    if (groups.empty()
        || primary_energy_eV < groups.front().primary_energy_eV
        || primary_energy_eV > groups.back().primary_energy_eV) {
        rrea::KineticFail("RREA conditional-CDF mean query is outside its range");
    }
    auto const upper = rrea::kinetic::lower_bound(
        groups.begin(), groups.end(), primary_energy_eV,
        [](RreaConditionalCdfGroupView const& group, amrex::Real energy) {
            return group.primary_energy_eV < energy;
        });
    if (upper == groups.begin()) return group_mean(*upper);
    if (upper == groups.end()) return group_mean(groups.back());
    if (upper->primary_energy_eV == primary_energy_eV) return group_mean(*upper);
    std::size_t const hi = static_cast<std::size_t>(rrea::kinetic::distance(groups.begin(), upper));
    std::size_t const lo = hi - 1U;
    amrex::Real const y0 = group_mean(groups[lo]);
    amrex::Real const y1 = group_mean(groups[hi]);
    amrex::Real const t =
        (std::log(primary_energy_eV) - std::log(groups[lo].primary_energy_eV))
        / (std::log(groups[hi].primary_energy_eV)
           - std::log(groups[lo].primary_energy_eV));
    return y0 > amrex::Real(0.0) && y1 > amrex::Real(0.0)
        ? std::exp(std::log(y0) + t * (std::log(y1) - std::log(y0)))
        : y0 + t * (y1 - y0);
}

RREA_HOST_DEVICE inline amrex::Real apply_electron_theta_debug_scale(RreaDebugTransportSensitivityOptions const& sensitivity, bool is_positron, amrex::Real cos_theta)
{
    if (is_positron) {
        return cos_theta;
    }
    amrex::Real const scale = sensitivity.electron_scattering_theta;
    if (!sensitivity.enable || scale == amrex::Real(1.0)) {
        return cos_theta;
    }
    double const clamped =
        rrea::kinetic::min(rrea::kinetic::max(static_cast<double>(cos_theta), -1.0), 1.0);
    return static_cast<amrex::Real>(
        std::cos(static_cast<double>(scale) * std::acos(clamped)));
}
} // kinetic_tables
RREA_HOST_DEVICE inline amrex::Real RreaInteractionTableView::Interpolate(amrex::Real energy_eV_query) const
{
    if (energy_eV.empty() || values.empty() || energy_eV.size() != values.size()
        || (mode != RreaInterpolationMode::Linear
            && log_energy.size() != energy_eV.size())) {
        rrea::KineticFail("invalid table interpolation query");
    }
    if (!std::isfinite(static_cast<double>(energy_eV_query))
        || energy_eV_query < energy_eV.front() || energy_eV_query > energy_eV.back()) {
        rrea::KineticFail("RREA interaction table '");
    }
    auto const upper = rrea::kinetic::lower_bound(energy_eV.begin(), energy_eV.end(), energy_eV_query);
    if (upper == energy_eV.begin()) {
        return values.front();
    }
    if (upper == energy_eV.end()) {
        return values.back();
    }
    std::size_t const hi = static_cast<std::size_t>(rrea::kinetic::distance(energy_eV.begin(), upper));
    std::size_t const lo = hi - 1U;
    if (energy_eV[hi] == energy_eV_query) {
        return values[hi];
    }
    if (mode == RreaInterpolationMode::Linear) {
        amrex::Real const t = (energy_eV_query - energy_eV[lo])
            / (energy_eV[hi] - energy_eV[lo]);
        return values[lo] + t * (values[hi] - values[lo]);
    }
    amrex::Real const t = (std::log(energy_eV_query) - log_energy[lo])
        / (log_energy[hi] - log_energy[lo]);
    bool const positive_bracket =
        values[lo] > amrex::Real(0.0) && values[hi] > amrex::Real(0.0);
    if (mode == RreaInterpolationMode::LogXLinear
        || (mode == RreaInterpolationMode::LogLogPositiveWithZeroThreshold
            && !positive_bracket)) {
        return values[lo] + t * (values[hi] - values[lo]);
    }
    if (!positive_bracket) {
        rrea::KineticFail("invalid table interpolation query");
    }
    return std::exp(log_values[lo] + t * (log_values[hi] - log_values[lo]));
}

RREA_HOST_DEVICE inline bool RreaPairChannelHasPhaseSpace(
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel) noexcept
{
    amrex::Real const threshold = channel == RreaPhotonChannel::PairTriplet
        ? kinetic_tables::kPairTripletThresholdEv : kinetic_tables::kPairNuclearThresholdEv;
    return std::isfinite(static_cast<double>(photon_energy_eV))
        && photon_energy_eV > threshold;
}

RREA_HOST_DEVICE inline RreaPairFinalState BuildRreaPairEmpiricalFinalState(
    RreaPairEmpiricalRecord const& record,
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real draw_pair_azimuth)
{
    RreaPairFinalState state;
    state.channel = channel;
    bool const triplet = channel == RreaPhotonChannel::PairTriplet;
    bool const nuclear = channel == RreaPhotonChannel::PairNuclear;
    if ((!triplet && !nuclear) || record.channel != channel
        || !RreaPairChannelHasPhaseSpace(photon_energy_eV, channel)) {
        return state;
    }

    int electron_slot = -1;
    int positron_slot = -1;
    int recoil_slot = -1;
    for (int slot = 0; slot < record.lepton_count; ++slot) {
        if (record.particles[slot].pdg == -11) {
            positron_slot = slot;
        } else if (record.particles[slot].pdg == 11) {
            if (electron_slot < 0) {
                electron_slot = slot;
            } else {
                recoil_slot = slot;
            }
        }
    }
    if (electron_slot < 0 || positron_slot < 0 || (triplet && recoil_slot < 0)) {
        rrea::KineticFail("schema-6 selected pair record has invalid lepton topology");
    }

    amrex::Real const available_kinetic_eV =
        photon_energy_eV - amrex::Real(2.0) * kinetic_tables::kElectronRestEnergyEv;
    amrex::Real const rotation = kinetic_tables::kTwoPi * kinetic_tables::clamp_unit(draw_pair_azimuth);
    amrex::Real const cos_rotation = std::cos(rotation);
    amrex::Real const sin_rotation = std::sin(rotation);
    auto rotated_direction = [&] (
        RreaPairEmpiricalParticle const& particle,
        amrex::Real& x,
        amrex::Real& y,
        amrex::Real& z) {
        x = cos_rotation * particle.dir_x - sin_rotation * particle.dir_y;
        y = sin_rotation * particle.dir_x + cos_rotation * particle.dir_y;
        z = particle.dir_z;
        kinetic_tables::normalize_local_direction(x, y, z);
    };

    state.valid = true;
    state.electron_energy_eV = available_kinetic_eV
        * record.particles[electron_slot].kinetic_fraction;
    state.positron_energy_eV = available_kinetic_eV
        * record.particles[positron_slot].kinetic_fraction;
    rotated_direction(
        record.particles[electron_slot], state.electron_dir_x,
        state.electron_dir_y, state.electron_dir_z);
    rotated_direction(
        record.particles[positron_slot], state.positron_dir_x,
        state.positron_dir_y, state.positron_dir_z);
    if (triplet) {
        state.creates_target_ion = true;
        state.has_recoil_electron = true;
        state.recoil_electron_energy_eV = available_kinetic_eV
            * record.particles[recoil_slot].kinetic_fraction;
        rotated_direction(
            record.particles[recoil_slot], state.recoil_dir_x,
            state.recoil_dir_y, state.recoil_dir_z);
    }

    amrex::Real const tolerance = amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * rrea::kinetic::max(amrex::Real(1.0), available_kinetic_eV);
    auto explicit_sum = [&]() {
        return state.electron_energy_eV + state.positron_energy_eV
            + state.recoil_electron_energy_eV;
    };
    amrex::Real explicit_kinetic_eV = explicit_sum();
    if (!std::isfinite(static_cast<double>(explicit_kinetic_eV))
        || state.electron_energy_eV < amrex::Real(0.0)
        || state.positron_energy_eV < amrex::Real(0.0)
        || state.recoil_electron_energy_eV < amrex::Real(0.0)) {
        rrea::KineticFail("schema-6 pair final-state record produced a negative/nonfinite energy");
    }
    if (explicit_kinetic_eV > available_kinetic_eV) {
        if (explicit_kinetic_eV - available_kinetic_eV > tolerance) {
            rrea::KineticFail("schema-6 pair final-state record failed energy closure");
        }
        amrex::Real const scale = available_kinetic_eV / explicit_kinetic_eV;
        state.electron_energy_eV *= scale;
        state.positron_energy_eV *= scale;
        state.recoil_electron_energy_eV *= scale;
        explicit_kinetic_eV = explicit_sum();
    }
    state.local_deposit_energy_eV = available_kinetic_eV - explicit_kinetic_eV;
    auto reduce_largest = [&] (amrex::Real correction) {
        std::array<amrex::Real*, 3> energies{{
            &state.electron_energy_eV,
            &state.positron_energy_eV,
            &state.recoil_electron_energy_eV,
        }};
        auto largest = rrea::kinetic::max_element(
            energies.begin(), energies.end(),
            [](amrex::Real const* lhs, amrex::Real const* rhs) {
                return *lhs < *rhs;
            });
        if (largest == energies.end() || **largest < correction) {
            rrea::KineticFail("schema-6 pair final-state cannot close nonnegative remainder");
        }
        **largest -= correction;
    };
    if (state.local_deposit_energy_eV < amrex::Real(0.0)) {
        amrex::Real const correction = -state.local_deposit_energy_eV;
        if (correction > tolerance) {
            rrea::KineticFail("schema-6 pair final-state exact remainder is negative");
        }
        reduce_largest(correction);
        state.local_deposit_energy_eV = amrex::Real(0.0);
    }
    amrex::Real accounted = explicit_sum() + state.local_deposit_energy_eV;
    state.energy_residual_eV = available_kinetic_eV - accounted;
    if (std::abs(state.energy_residual_eV) > tolerance) {
        rrea::KineticFail("schema-6 pair final-state record failed energy closure");
    }
    if (state.energy_residual_eV >= amrex::Real(0.0)) {
        state.local_deposit_energy_eV += state.energy_residual_eV;
    } else if (state.local_deposit_energy_eV >= -state.energy_residual_eV) {
        state.local_deposit_energy_eV += state.energy_residual_eV;
    } else {
        reduce_largest(-state.energy_residual_eV);
    }
    accounted = explicit_sum() + state.local_deposit_energy_eV;
    if (state.electron_energy_eV < amrex::Real(0.0)
        || state.positron_energy_eV < amrex::Real(0.0)
        || state.recoil_electron_energy_eV < amrex::Real(0.0)
        || state.local_deposit_energy_eV < amrex::Real(0.0)
        || std::abs(available_kinetic_eV - accounted) > tolerance) {
        rrea::KineticFail("schema-6 pair final-state nonnegative closure correction failed");
    }
    state.energy_residual_eV = amrex::Real(0.0);
    return state;
}

struct RreaInteractionTablesView {

    RREA_HOST_DEVICE bool Loaded() const noexcept {return m_loaded;}
    RREA_HOST_DEVICE int SchemaVersion() const noexcept {return m_loaded ? 6 : 0;}
    RREA_HOST_DEVICE RreaCertifiedEnergyRange ElectronCertifiedEnergyRangeEv() const noexcept {return m_charged_certified_range;}
    RREA_HOST_DEVICE RreaCertifiedEnergyRange PositronCertifiedEnergyRangeEv() const noexcept {return m_charged_certified_range;}
    RREA_HOST_DEVICE RreaCertifiedEnergyRange PhotonCertifiedEnergyRangeEv() const noexcept {return m_photon_certified_range;}
    RREA_HOST_DEVICE amrex::Real GsFewCollisionThreshold() const noexcept {return m_gs_few_collision_threshold;}
    RREA_HOST_DEVICE amrex::Real HardMollerSecondaryThresholdEv() const noexcept {return m_hard_moller_secondary_threshold_eV;}
    RREA_HOST_DEVICE amrex::Real BremsAbsoluteCutoffEv() const noexcept {return m_brems_absolute_cutoff_eV;}
    RREA_HOST_DEVICE amrex::Real WAirEvPerIonPair() const noexcept {return static_cast<amrex::Real>(mean_energy_per_ion_pair_air_eV);}
    RREA_HOST_DEVICE RreaDebugTransportSensitivityOptions const& DebugOptions() const noexcept {return m_debug;}
    RREA_HOST_DEVICE RreaInteractionTableView const& Require(RreaTable id) const noexcept {return m_hot[static_cast<std::size_t>(id)];}

    RREA_HOST_DEVICE [[nodiscard]] amrex::Real ReferenceDensityKgM3() const noexcept
    {
        return m_reference_density_kg_m3;
    }
    RREA_HOST_DEVICE void RequireSchema6SamplerEnergy(
    char const* sampler,
    char const* species,
    RreaCertifiedEnergyRange const& certified_range,
    amrex::Real energy_eV,
    bool allow_exact_zero = false) const;
    RREA_HOST_DEVICE RreaPhotonProcessInverseLengths PhotonProcessInverseLengths(
    amrex::Real energy_eV, amrex::Real density_ratio) const;
    RREA_HOST_DEVICE RreaElectronTransportCoefficients ElectronTransportCoefficients(
    amrex::Real energy_eV, amrex::Real density_ratio) const;
    RREA_HOST_DEVICE RreaPositronTransportCoefficients PositronTransportCoefficients(
    amrex::Real energy_eV, amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real ElectronCollisionLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real ElectronHardMollerMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real ElectronHardMollerEnergyLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real ElectronElasticTransportMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real ElasticMeanFreePath(
    bool is_positron, amrex::Real energy_eV, amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real ScreeningEta(bool is_positron, amrex::Real energy_eV) const;
    RREA_HOST_DEVICE amrex::Real SampleGsCosTheta(
    bool is_positron, amrex::Real energy_eV, amrex::Real path_n, amrex::Real cdf_u) const;
    RREA_HOST_DEVICE amrex::Real SampleSingleElasticCosTheta(
    bool is_positron, amrex::Real energy_eV, amrex::Real cdf_u) const;
    RREA_HOST_DEVICE amrex::Real SamplePositronBremsstrahlungPhotonEnergy(
    amrex::Real primary_energy_eV, amrex::Real cdf_u) const;
    RREA_HOST_DEVICE amrex::Real SampleMollerSecondaryEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const;
    RREA_HOST_DEVICE amrex::Real ElectronSoftRadiativeDrag(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real ElectronBremsstrahlungMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real ElectronTrackedBremsEnergyLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real SampleBremsstrahlungPhotonEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const;
    RREA_HOST_DEVICE amrex::Real SampleBremsstrahlungCosTheta(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const;
    RREA_HOST_DEVICE amrex::Real PhotonFeedbackMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE RreaPhotonChannel SamplePhotonChannel(
    amrex::Real energy_eV,
    amrex::Real cdf_u) const;
    RREA_HOST_DEVICE RreaComptonFinalState SampleComptonKleinNishinaFinalState(
    amrex::Real primary_energy_eV,
    amrex::Real draw_mu,
    amrex::Real draw_accept) const;
    RREA_HOST_DEVICE RreaPhotoelectricNumericState SamplePhotoelectricFinalState(
    amrex::Real photon_energy_eV,
    amrex::Real cdf_u) const;
    RREA_HOST_DEVICE amrex::Real PhotoelectricElectronEnergy(
    amrex::Real photon_energy_eV) const;
    RREA_HOST_DEVICE amrex::Real SamplePhotoelectronCosTheta(
    amrex::Real photoelectron_energy_eV,
    amrex::Real draw_polar) const;
    RREA_HOST_DEVICE RreaPairFinalState SamplePairFinalState(
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real draw_invariant,
    amrex::Real draw_pair_polar,
    amrex::Real draw_pair_azimuth,
    amrex::Real draw_lepton_polar,
    amrex::Real draw_lepton_azimuth) const;
    RREA_HOST_DEVICE amrex::Real PositronCollisionLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real PositronSoftRadiativeDrag(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real PositronBremsstrahlungMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real PositronHardBhabhaMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE amrex::Real SamplePositronBhabhaSecondaryEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const;
    RREA_HOST_DEVICE amrex::Real PositronAnnihilationMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    RREA_HOST_DEVICE RreaPositronAnnihilationFinalState SamplePositronAnnihilationFinalState(
    amrex::Real positron_kinetic_energy_eV,
    amrex::Real draw_energy,
    amrex::Real draw_azimuth) const;
    RREA_HOST_DEVICE amrex::Real PositronElasticTransportMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const;
    bool m_loaded=false;
    amrex::Real m_hard_moller_secondary_threshold_eV=0, m_brems_absolute_cutoff_eV=0, m_reference_density_kg_m3=0;
    RreaCertifiedEnergyRange m_charged_certified_range, m_photon_certified_range;
    RreaDebugTransportSensitivityOptions m_debug;
    std::array<RreaInteractionTableView,static_cast<std::size_t>(RreaTable::Count)> m_hot{};
    KineticSpan<amrex::Real const> m_hard_moller_mean_secondary_eV;
    KineticSpan<RreaMollerCdfGroupView const> m_moller_cdf;
    std::array<KineticSpan<RreaConditionalCdfGroupView const>,3> m_conditional_cdfs{};
    KineticSpan<RreaPhotoelectricSubshellCdfGroupView const> m_photoelectric_subshell_cdf;
    RreaElasticMfpTableView m_electron_elastic_mfp,m_positron_elastic_mfp;
    RreaGsAngleTableView m_electron_gs_angle,m_positron_gs_angle,m_electron_wentzel_angle,m_positron_wentzel_angle;
    KineticSpan<RreaPairEmpiricalBlockView const> m_pair_empirical_blocks;
    amrex::Real m_gs_few_collision_threshold=10, m_gs_isotropic_above_n=1e5, m_gs_wentzel_transition_energy_eV=0;

};

RREA_HOST_DEVICE inline void RreaInteractionTablesView::RequireSchema6SamplerEnergy(
    char const* sampler,
    char const* species,
    RreaCertifiedEnergyRange const& certified_range,
    amrex::Real energy_eV,
    bool allow_exact_zero) const
{
    (void)sampler; (void)species;
    if (allow_exact_zero && energy_eV == amrex::Real(0.0)) {
        return;
    }
    if (!std::isfinite(static_cast<double>(energy_eV))
        || !certified_range.Contains(energy_eV)) {
        rrea::KineticFail(": schema-6 ");
    }
}

RREA_HOST_DEVICE inline RreaPhotonProcessInverseLengths RreaInteractionTablesView::PhotonProcessInverseLengths(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    if (!std::isfinite(static_cast<double>(density_ratio))
        || density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("PhotonProcessInverseLengths density_ratio must be finite and positive");
    }
    if (!m_photon_certified_range.Contains(energy_eV)) {
        rrea::KineticFail("schema-6 photon energy ");
    }
    return RreaPhotonProcessInverseLengths{
        Require(RreaTable::PhotonCompton).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PhotonPhotoelectric).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PhotonPairNuclear).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PhotonPairTriplet).Interpolate(energy_eV)
            * density_ratio};
}

RREA_HOST_DEVICE inline RreaElectronTransportCoefficients RreaInteractionTablesView::ElectronTransportCoefficients(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    if (!std::isfinite(static_cast<double>(density_ratio))
        || density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("ElectronTransportCoefficients density_ratio must be finite and positive");
    }
    if (!m_charged_certified_range.Contains(energy_eV)) {
        rrea::KineticFail("schema-6 electron energy ");
    }
    return RreaElectronTransportCoefficients{
        Require(RreaTable::ElectronCollisionStopping).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::ElectronSoftRadiativeStopping).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::ElectronHardMollerInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::ElectronTrackedBremsInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::ElectronElasticTransportInverseLength).Interpolate(energy_eV)
            * density_ratio};
}

RREA_HOST_DEVICE inline RreaPositronTransportCoefficients RreaInteractionTablesView::PositronTransportCoefficients(
    amrex::Real energy_eV, amrex::Real density_ratio) const
{
    if (!std::isfinite(static_cast<double>(density_ratio))
        || density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("PositronTransportCoefficients density_ratio must be finite and positive");
    }
    if (!m_charged_certified_range.Contains(energy_eV)) {
        rrea::KineticFail("schema-6 positron energy ");
    }
    return RreaPositronTransportCoefficients{
        Require(RreaTable::PositronCollisionStopping).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronSoftRadiativeStopping).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronHardBhabhaInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronTrackedBremsInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronElasticTransportInverseLength).Interpolate(energy_eV)
            * density_ratio,
        Require(RreaTable::PositronAnnihilationInverseLength).Interpolate(energy_eV)
            * density_ratio};
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ElectronCollisionLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    amrex::Real const loss_stp_eV_per_m =
        Require(RreaTable::ElectronCollisionStopping).Interpolate(energy_eV);
    return loss_stp_eV_per_m * density_ratio;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ElectronHardMollerMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    // Validate the certified energy before the zero-phase-space guard; an
    // out-of-range query must never be disguised as an infinite MFP.
    amrex::Real const inverse_m_stp =
        Require(RreaTable::ElectronHardMollerInverseLength).Interpolate(energy_eV);
    // The rate table is exactly zero at 2*Tcut and authoritative immediately
    // above it.  Do not widen that zero-rate point into an unmodelled band.
    if (m_hard_moller_secondary_threshold_eV > amrex::Real(0.0)
        && energy_eV <= amrex::Real(2.0) * m_hard_moller_secondary_threshold_eV) {
        return amrex::Real(1.0e99);
    }
    return kinetic_tables::mean_free_path_from_inverse_length(inverse_m_stp * density_ratio);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ElectronHardMollerEnergyLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    // Mean drag from the DISCRETE delta-ray channel: <T> / lambda_hardMoller.
    // The parent's actual loss is realized stochastically by sampled Moller
    // events, so this must never be added to the continuous drag that transport
    // applies -- that would double-count.  It exists because restricted
    // stopping power alone is not the break-even criterion: at the 1.2 MeV
    // minimum it is 180.5 kV/m against a true total of 199.4, and a runaway
    // gate built on the restricted value fires about 10 percent too easily.
    auto const& rate = Require(RreaTable::ElectronHardMollerInverseLength);
    amrex::Real const inverse_m_stp = rate.Interpolate(energy_eV);
    if (!(inverse_m_stp > amrex::Real(0.0))
        || m_hard_moller_mean_secondary_eV.size() != rate.energy_eV.size()) {
        return amrex::Real(0.0);
    }
    amrex::Real const mean_secondary_eV = kinetic_tables::interp_semilogx_strict(
        rate.energy_eV, m_hard_moller_mean_secondary_eV, energy_eV);
    if (!(mean_secondary_eV > amrex::Real(0.0))) {
        return amrex::Real(0.0);
    }
    return mean_secondary_eV * inverse_m_stp * density_ratio;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ElectronElasticTransportMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return kinetic_tables::mean_free_path_from_inverse_length(
        ElectronTransportCoefficients(energy_eV, density_ratio)
            .elastic_transport_inverse_m);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ElasticMeanFreePath(
    bool is_positron, amrex::Real energy_eV, amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    auto const& tab = is_positron ? m_positron_elastic_mfp : m_electron_elastic_mfp;
    if (tab.energy_eV.empty()) {
        rrea::KineticFail("ElasticMeanFreePath: elastic collision-MFP table not loaded");
    }
    // This is the collision MFP used in n=s/lambda_el by the few-collision and
    // conditional-angle samplers. It is deliberately distinct from the
    // unified coefficient table's elastic *transport* inverse length (1/lambda1).
    // STP collision MFP shortens with density (lambda = lambda_stp/rho_ratio).
    return kinetic_tables::interp_semilogx_strict(tab.energy_eV, tab.elastic_mfp_m_stp, energy_eV)
        / density_ratio;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ScreeningEta(bool is_positron, amrex::Real energy_eV) const
{
    auto const& tab = is_positron ? m_positron_elastic_mfp : m_electron_elastic_mfp;
    if (tab.energy_eV.empty()) {
        rrea::KineticFail("ScreeningEta: elastic parameter table not loaded");
    }
    return kinetic_tables::interp_semilogx_strict(tab.energy_eV, tab.screening_eta, energy_eV);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::SampleGsCosTheta(
    bool is_positron, amrex::Real energy_eV, amrex::Real path_n, amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleGsCosTheta",
        is_positron ? "positron" : "electron",
        m_charged_certified_range,
        energy_eV);
    if (!std::isfinite(static_cast<double>(cdf_u))
        || cdf_u < amrex::Real(0.0)
        || cdf_u > amrex::Real(1.0)) {
        rrea::KineticFail("SampleGsCosTheta: schema-6 CDF query must be in [0,1]");
    }
    bool const use_gs = energy_eV <= m_gs_wentzel_transition_energy_eV;
    RreaGsAngleTableView const* tab = nullptr;
    if (is_positron) {
        tab = use_gs ? &m_positron_gs_angle : &m_positron_wentzel_angle;
    } else {
        tab = use_gs ? &m_electron_gs_angle : &m_electron_wentzel_angle;
    }
    if (tab == nullptr || tab->energy_eV.empty()) {
        rrea::KineticFail("SampleGsCosTheta: selected elastic angle table is not loaded");
    }
    if (!std::isfinite(static_cast<double>(energy_eV))
        || energy_eV < tab->energy_eV.front()
        || energy_eV > tab->energy_eV.back()) {
        rrea::KineticFail("SampleGsCosTheta: energy is outside the selected model branch");
    }
    amrex::Real const u = cdf_u;
    if (!std::isfinite(static_cast<double>(path_n))
        || path_n < m_gs_few_collision_threshold) {
        rrea::KineticFail(
            "SampleGsCosTheta: path n is below the multiple-scattering branch "
            "or is non-finite");
    }
    if (path_n >= m_gs_isotropic_above_n) {
        return amrex::Real(2.0) * u - amrex::Real(1.0);  // isotropic
    }
    amrex::Real const cos_theta = kinetic_tables::sample_gs_angle_cos(*tab, energy_eV, path_n, u);
    return kinetic_tables::apply_electron_theta_debug_scale(m_debug, is_positron, cos_theta);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::SampleSingleElasticCosTheta(
    bool is_positron, amrex::Real energy_eV, amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleSingleElasticCosTheta",
        is_positron ? "positron" : "electron",
        m_charged_certified_range,
        energy_eV);
    // Analytic screened-Rutherford inverse-CDF draw.
    amrex::Real const eta = ScreeningEta(is_positron, energy_eV);
    amrex::Real const u = rrea::kinetic::min(rrea::kinetic::max(cdf_u, amrex::Real(0.0)), amrex::Real(1.0) - amrex::Real(1.0e-15));
    amrex::Real const cos_theta =
        amrex::Real(1.0) - amrex::Real(2.0) * eta * u / (amrex::Real(1.0) + eta - u);
    return kinetic_tables::apply_electron_theta_debug_scale(m_debug, is_positron, cos_theta);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::SamplePositronBremsstrahlungPhotonEnergy(
    amrex::Real primary_energy_eV, amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SamplePositronBremsstrahlungPhotonEnergy",
        "positron",
        m_charged_certified_range,
        primary_energy_eV);
    auto const* it = &m_conditional_cdfs[1];
    if (it->empty()) {
        rrea::KineticFail("SamplePositronBremsstrahlungPhotonEnergy: positron brems CDF not loaded");
    }
    return kinetic_tables::sample_conditional_cdf(*it, primary_energy_eV, cdf_u, /*log_interp=*/true);
}

// ===== Hard Moller / Bhabha secondary-energy samplers =====
// Conditional CDFs from the Geant4-generated bundle.  Endpoint kinematics
// (data, not bounds): Moller spans [T_cut, E/2] -- identical particles, the
// faster lepton is primary by convention, so the channel opens only at
// E > 2 T_cut; Bhabha spans [T_cut, E] (distinguishable) and opens at
// E > T_cut.  The zero-phase-space band below each opening carries an
// infinite MFP in the rate tables, not a clamp.
RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::SampleMollerSecondaryEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleMollerSecondaryEnergy",
        "electron",
        m_charged_certified_range,
        primary_energy_eV);
    if (m_moller_cdf.empty()) {
        rrea::KineticFail("Moller CDF table is not loaded");
    }
    if (!std::isfinite(static_cast<double>(primary_energy_eV))
        || primary_energy_eV < m_moller_cdf.front().primary_energy_eV
        || primary_energy_eV > m_moller_cdf.back().primary_energy_eV) {
        rrea::KineticFail("Moller CDF primary energy ");
    }
    cdf_u = rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), cdf_u));

    auto const sample_group = [cdf_u](RreaMollerCdfGroupView const& group) {
        auto const upper = rrea::kinetic::lower_bound(group.cdf_u.begin(), group.cdf_u.end(), cdf_u);
        if (upper == group.cdf_u.begin()) {
            return group.secondary_energy_eV.front();
        }
        if (upper == group.cdf_u.end()) {
            return group.secondary_energy_eV.back();
        }
        std::size_t const hi =
            static_cast<std::size_t>(rrea::kinetic::distance(group.cdf_u.begin(), upper));
        std::size_t const lo = hi - 1U;
        amrex::Real const denom = group.cdf_u[hi] - group.cdf_u[lo];
        amrex::Real const t = denom > amrex::Real(0.0)
            ? (cdf_u - group.cdf_u[lo]) / denom
            : amrex::Real(0.0);
        return group.secondary_energy_eV[lo]
            + t * (group.secondary_energy_eV[hi] - group.secondary_energy_eV[lo]);
    };

    auto const upper = rrea::kinetic::lower_bound(
        m_moller_cdf.begin(),
        m_moller_cdf.end(),
        primary_energy_eV,
        [](RreaMollerCdfGroupView const& group, amrex::Real energy) {
            return group.primary_energy_eV < energy;
        });
    if (upper == m_moller_cdf.begin()) {
        return sample_group(*upper);
    }
    if (upper == m_moller_cdf.end()) {
        return sample_group(m_moller_cdf.back());
    }
    std::size_t const hi =
        static_cast<std::size_t>(rrea::kinetic::distance(m_moller_cdf.begin(), upper));
    std::size_t const lo = hi - 1U;
    if (m_moller_cdf[hi].primary_energy_eV == primary_energy_eV) {
        return sample_group(m_moller_cdf[hi]);
    }
    amrex::Real const y0 = sample_group(m_moller_cdf[lo]);
    amrex::Real const y1 = sample_group(m_moller_cdf[hi]);
    amrex::Real const x0 = std::log(m_moller_cdf[lo].primary_energy_eV);
    amrex::Real const x1 = std::log(m_moller_cdf[hi].primary_energy_eV);
    amrex::Real const t = (std::log(primary_energy_eV) - x0) / (x1 - x0);
    return std::exp(std::log(y0) + t * (std::log(y1) - std::log(y0)));
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ElectronSoftRadiativeDrag(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return Require(RreaTable::ElectronSoftRadiativeStopping).Interpolate(energy_eV)
        * density_ratio;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ElectronBremsstrahlungMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return kinetic_tables::mean_free_path_from_inverse_length(
        Require(RreaTable::ElectronTrackedBremsInverseLength).Interpolate(energy_eV)
            * density_ratio);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::ElectronTrackedBremsEnergyLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    auto const* it = &m_conditional_cdfs[0];
    if (it->empty()) {
        rrea::KineticFail("brems_photon_energy_cdf_air is not loaded");
    }
    amrex::Real const inverse_length =
        Require(RreaTable::ElectronTrackedBremsInverseLength).Interpolate(energy_eV)
        * density_ratio;
    if (!(inverse_length > amrex::Real(0.0))) {
        return amrex::Real(0.0);
    }
    return inverse_length * kinetic_tables::mean_conditional_cdf(*it, energy_eV);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::SampleBremsstrahlungPhotonEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleBremsstrahlungPhotonEnergy",
        "electron",
        m_charged_certified_range,
        primary_energy_eV);
    auto const* it = &m_conditional_cdfs[0];
    if (it->empty()) {
        rrea::KineticFail("brems_photon_energy_cdf_air is not loaded");
    }
    return kinetic_tables::sample_conditional_cdf(*it, primary_energy_eV, cdf_u, true);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::SampleBremsstrahlungCosTheta(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SampleBremsstrahlungCosTheta",
        "charged-lepton",
        m_charged_certified_range,
        primary_energy_eV);
    // Bounded ModifiedTsai (Geant4 10.7.4 G4ModifiedTsai::SampleCosTheta;
    // Urban, Geant3 Phys211, derived from Tsai, Rev. Mod. Phys. 49, 421).
    // F(u) = .25 G(a*u) + .75 G(3*a*u), G(x)=1-(1+x)exp(-x), a=.625.
    // Condition the whole mixture on u <= 2*gamma, then use
    // cos(theta)=1-2*(u/(2*gamma))^2. Inverting p*F(2*gamma) is equivalent
    // to Geant4's rejection of out-of-support draws, with one uniform input.
    cdf_u = rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), cdf_u));
    if (cdf_u == amrex::Real(0.0)) { return amrex::Real(1.0); }
    if (cdf_u == amrex::Real(1.0)) { return amrex::Real(-1.0); }
    amrex::Real constexpr tsai_a = amrex::Real(0.625);
    auto gamma2_cdf = [](amrex::Real x) {
        if (x < amrex::Real(1.0e-3)) {
            return x*x * (amrex::Real(0.5) + x * (amrex::Real(-1.0/3.0)
                + x * (amrex::Real(1.0/8.0) + x * (amrex::Real(-1.0/30.0)
                + x * amrex::Real(1.0/144.0)))));
        }
        return -std::expm1(-x) - x * std::exp(-x);
    };
    auto mixture_cdf = [&](amrex::Real u) {
        return amrex::Real(0.25) * gamma2_cdf(tsai_a * u)
            + amrex::Real(0.75) * gamma2_cdf(amrex::Real(3.0) * tsai_a * u);
    };
    amrex::Real const u_max = amrex::Real(2.0)
        * (amrex::Real(1.0) + primary_energy_eV / kinetic_tables::kElectronRestEnergyEv);
    amrex::Real const target = cdf_u * mixture_cdf(u_max);
    amrex::Real lo = amrex::Real(0.0);
    amrex::Real hi = u_max;
    for (int iter = 0; iter < 60; ++iter) {
        amrex::Real const mid = amrex::Real(0.5) * (lo + hi);
        if (mixture_cdf(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    amrex::Real const fraction = amrex::Real(0.5) * (lo + hi) / u_max;
    return amrex::Real(1.0) - amrex::Real(2.0) * fraction * fraction;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::PhotonFeedbackMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return kinetic_tables::mean_free_path_from_inverse_length(
        PhotonProcessInverseLengths(energy_eV, density_ratio).TotalPerM());
}

// ===== Photon channel selection =====
// Partitions ONE uniform draw by the same per-channel inverse lengths that
// form the total photon MFP (photoelectric / Compton / pair nuclear / pair
// triplet), so selection and hazard rate cannot disagree; the density ratio
// cancels in the branching fractions, which is why none is passed here.
RREA_HOST_DEVICE inline RreaPhotonChannel RreaInteractionTablesView::SamplePhotonChannel(
    amrex::Real energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SamplePhotonChannel", "photon", m_photon_certified_range, energy_eV);
    cdf_u = rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), cdf_u));
    auto const rates = PhotonProcessInverseLengths(
        energy_eV, amrex::Real(1.0));
    amrex::Real const total = rates.TotalPerM();
    if (!(total > amrex::Real(0.0))
        || !std::isfinite(static_cast<double>(total))) {
        rrea::KineticFail("schema-6 photon process inverse lengths sum to zero/non-finite");
    }
    amrex::Real const target = cdf_u * total;
    if (target < rates.compton_per_m) {
        return RreaPhotonChannel::Compton;
    }
    if (target < rates.compton_per_m + rates.photoelectric_per_m) {
        return RreaPhotonChannel::Photoelectric;
    }
    if (target < rates.compton_per_m + rates.photoelectric_per_m
            + rates.pair_nuclear_per_m) {
        return RreaPhotonChannel::PairNuclear;
    }
    return RreaPhotonChannel::PairTriplet;
}

RREA_HOST_DEVICE inline RreaComptonFinalState RreaInteractionTablesView::SampleComptonKleinNishinaFinalState(
    amrex::Real primary_energy_eV,
    amrex::Real draw_mu,
    amrex::Real draw_accept) const
{
    RequireSchema6SamplerEnergy(
        "SampleComptonKleinNishinaFinalState",
        "photon",
        m_photon_certified_range,
        primary_energy_eV);
    if (primary_energy_eV <= amrex::Real(0.0)) {
        rrea::KineticFail("Compton sampler requires positive photon energy");
    }
    // Exact free-electron Klein-Nishina final state via the Butcher-Messel
    // rejection scheme used by G4KleinNishinaCompton.
    //
    // The runtime hands this sampler two coupled uniforms (channels
    // ComptonRecoil=53, ComptonScatter=54).  The accept/reject loop needs
    // more: round 0 consumes the two pre-drawn uniforms directly, while the
    // accept draw and later rounds use a deterministic stream seeded from them.
    //
    // Deliberate rate/final-state mismatch: photon_compton_inverse_length_air
    // carries Geant4's incoherent scattering function (-4.4% vs free KN at
    // 50 keV). MFP and channel probabilities retain that correction, but the
    // sampled energy-angle distribution assumes a free stationary electron.
    // Per-event energy is conserved; bound-electron kinematics are omitted.
    draw_mu = rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), draw_mu));
    draw_accept = rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), draw_accept));
    RreaUniformStream stream(static_cast<double>(draw_mu), static_cast<double>(draw_accept));

    amrex::Real const alpha = primary_energy_eV / kinetic_tables::kElectronRestEnergyEv;  // E0 / mc^2
    amrex::Real const eps0 = amrex::Real(1.0) / (amrex::Real(1.0) + amrex::Real(2.0) * alpha);
    amrex::Real const eps0sq = eps0 * eps0;
    amrex::Real const a1 = -std::log(eps0);                       // ln(1 + 2 alpha)
    amrex::Real const a2 = amrex::Real(0.5) * (amrex::Real(1.0) - eps0sq);
    amrex::Real epsilon = eps0;   // scattered / incident photon energy ratio
    amrex::Real onecost = amrex::Real(0.0);
    int constexpr max_iter = 128;
    for (int iter = 0; iter < max_iter; ++iter) {
        amrex::Real const u1 = (iter == 0) ? draw_mu : static_cast<amrex::Real>(stream.Next());
        amrex::Real const u2 = (iter == 0) ? draw_accept : static_cast<amrex::Real>(stream.Next());
        amrex::Real epsilonsq;
        if (a1 > (a1 + a2) * u1) {
            epsilon = std::exp(-a1 * u2);          // 1/eps sampled ~ 1/eps  (eps in [eps0,1])
            epsilonsq = epsilon * epsilon;
        } else {
            epsilonsq = eps0sq + (amrex::Real(1.0) - eps0sq) * u2;  // eps sampled ~ eps
            epsilon = std::sqrt(epsilonsq);
        }
        onecost = (amrex::Real(1.0) - epsilon) / (epsilon * alpha);  // 1 - cos(theta)
        amrex::Real const sint2 = onecost * (amrex::Real(2.0) - onecost);
        amrex::Real const greject =
            amrex::Real(1.0) - epsilon * sint2 / (amrex::Real(1.0) + epsilonsq);
        if (greject >= static_cast<amrex::Real>(stream.Next())) {
            break;
        }
    }
    amrex::Real mu = amrex::Real(1.0) - onecost;
    mu = rrea::kinetic::max(amrex::Real(-1.0), rrea::kinetic::min(amrex::Real(1.0), mu));
    amrex::Real const scattered = epsilon * primary_energy_eV;
    RreaComptonFinalState state;
    state.scattered_photon_energy_eV = rrea::kinetic::max(amrex::Real(0.0), scattered);
    state.recoil_electron_energy_eV = rrea::kinetic::max(amrex::Real(0.0), primary_energy_eV - state.scattered_photon_energy_eV);
    state.cos_theta = mu;
    return state;
}

// ===== Photoelectric final state =====
// Subshell selection by tabulated per-shell probabilities (summing to one at
// load), photoelectron energy = E_gamma - E_binding of the selected shell,
// with the residual binding booked through atomic relaxation; the emission
// angle is Sauter-Gavrila (its own documented sampler below).
RREA_HOST_DEVICE inline RreaPhotoelectricNumericState RreaInteractionTablesView::SamplePhotoelectricFinalState(
    amrex::Real photon_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SamplePhotoelectricFinalState",
        "photon",
        m_photon_certified_range,
        photon_energy_eV);
    if (photon_energy_eV <= amrex::Real(0.0)) {
        rrea::KineticFail("Photoelectric sampler requires positive photon energy");
    }
    if (m_photoelectric_subshell_cdf.empty()) {
        rrea::KineticFail("Schema-6 photoelectric subshell CDF table is not loaded");
    }
    if (photon_energy_eV < m_photoelectric_subshell_cdf.front().primary_energy_eV
        || photon_energy_eV > m_photoelectric_subshell_cdf.back().primary_energy_eV) {
        rrea::KineticFail("Photoelectric subshell CDF primary energy out of range");
    }
    cdf_u = rrea::kinetic::max(amrex::Real(0.0), rrea::kinetic::min(amrex::Real(1.0), cdf_u));
    auto const upper = rrea::kinetic::lower_bound(
        m_photoelectric_subshell_cdf.begin(),
        m_photoelectric_subshell_cdf.end(),
        photon_energy_eV,
        [](RreaPhotoelectricSubshellCdfGroupView const& group, amrex::Real energy) {
            return group.primary_energy_eV < energy;
        });
    RreaPhotoelectricSubshellCdfGroupView const* group = nullptr;
    if (upper == m_photoelectric_subshell_cdf.begin()) {
        group = &(*upper);
    } else if (upper == m_photoelectric_subshell_cdf.end()) {
        group = &m_photoelectric_subshell_cdf.back();
    } else if (upper->primary_energy_eV == photon_energy_eV) {
        // At duplicated XCOM edge energies this intentionally picks the low-side
        // branch. Queries just above the edge use the high-side duplicate through
        // the previous-group rule below.
        group = &(*upper);
    } else {
        group = &(*(upper - 1));
    }
    if (group == nullptr || group->rows.empty()) {
        rrea::KineticFail("Photoelectric subshell CDF group is malformed");
    }
    auto const row_it = rrea::kinetic::lower_bound(
        group->rows.begin(),
        group->rows.end(),
        cdf_u,
        [](RreaPhotoelectricSubshellCdfRowView const& row, amrex::Real u) {
            return row.cdf_u < u;
        });
    RreaPhotoelectricSubshellCdfRowView const& row =
        row_it == group->rows.end() ? group->rows.back() : *row_it;
    if (row.binding_energy_eV > photon_energy_eV) {
        rrea::KineticFail("Photoelectric shell binding energy exceeds photon energy");
    }
    RreaPhotoelectricNumericState state;
    state.binding_energy_eV = row.binding_energy_eV;
    state.relaxation_energy_eV = row.relaxation_energy_eV;
    state.photoelectron_energy_eV = rrea::kinetic::max(amrex::Real(0.0), photon_energy_eV - row.binding_energy_eV);
    state.element_Z = row.element_Z;
    state.group_index = static_cast<std::size_t>(group - m_photoelectric_subshell_cdf.data());
    state.row_index = static_cast<std::size_t>(&row - group->rows.data());
    return state;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::PhotoelectricElectronEnergy(
    amrex::Real photon_energy_eV) const
{
    return SamplePhotoelectricFinalState(photon_energy_eV, amrex::Real(0.5)).photoelectron_energy_eV;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::SamplePhotoelectronCosTheta(
    amrex::Real photoelectron_energy_eV,
    amrex::Real draw_polar) const
{
    RequireSchema6SamplerEnergy(
        "SamplePhotoelectronCosTheta",
        "electron",
        m_charged_certified_range,
        photoelectron_energy_eV,
        /*allow_exact_zero=*/true);
    // Sample the polar cos(theta) of the photoelectron relative to the incident
    // photon momentum from the relativistic Sauter-Gavrila K-shell distribution
    // (the model G4LivermorePhotoElectricModel / SauterGavrila uses in option4):
    //   dsigma/dmu ~ (1 - mu^2) / (1 - beta mu)^4 * [1 + b (1 - beta mu)]
    // with mu = cos(theta) and b = 0.5 gamma (gamma-1)(gamma-2).  In
    // t = 1-beta*mu, the target density (apart from a positive constant) is
    //
    //   C t^-4 + (2+bC) t^-3 + (2b-1) t^-2 - b t^-1,
    //   C = beta^2-1.
    //
    // Its analytic primitive is monotone on [1-beta,1+beta], so a bracketed
    // inversion consumes exactly one PhotoelectricAngle uniform and avoids
    // bounded-rejection bias at high energy.
    rrea::RreaKineticWide const u = static_cast<rrea::RreaKineticWide>(kinetic_tables::clamp_unit(draw_polar));
    if (u <= rrea::RreaKineticWide(0.0)) {
        return amrex::Real(-1.0);
    }
    if (u >= rrea::RreaKineticWide(1.0)) {
        return amrex::Real(1.0);
    }

    rrea::RreaKineticWide const gamma = rrea::RreaKineticWide(1.0)
        + static_cast<rrea::RreaKineticWide>(photoelectron_energy_eV)
            / static_cast<rrea::RreaKineticWide>(kinetic_tables::kElectronRestEnergyEv);
    rrea::RreaKineticWide const inv_gamma_sq = rrea::RreaKineticWide(1.0) / (gamma * gamma);
    rrea::RreaKineticWide const beta = std::sqrt(rrea::kinetic::max(rrea::RreaKineticWide(0.0), rrea::RreaKineticWide(1.0) - inv_gamma_sq));

    if (beta < rrea::RreaKineticWide(1.0e-4)) {
        // The exact beta->0 limit is f(mu) = 3/4 (1-mu^2), with
        // F(mu) = (2+3mu-mu^3)/4. Invert it monotonically to preserve that
        // limiting distribution.
        rrea::RreaKineticWide lo = -rrea::RreaKineticWide(1.0);
        rrea::RreaKineticWide hi = rrea::RreaKineticWide(1.0);
        for (int iter = 0; iter < 96; ++iter) {
            rrea::RreaKineticWide const mid = rrea::RreaKineticWide(0.5) * (lo + hi);
            if (mid == lo || mid == hi) {
                break;
            }
            rrea::RreaKineticWide const cdf = (rrea::RreaKineticWide(2.0) + mid * (rrea::RreaKineticWide(3.0) - mid * mid)) / rrea::RreaKineticWide(4.0);
            if (cdf < u) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return static_cast<amrex::Real>(rrea::RreaKineticWide(0.5) * (lo + hi));
    }

    rrea::RreaKineticWide const b = rrea::RreaKineticWide(0.5) * gamma * (gamma - rrea::RreaKineticWide(1.0)) * (gamma - rrea::RreaKineticWide(2.0));
    rrea::RreaKineticWide const c = -inv_gamma_sq;
    // Compute 1-beta without subtractive cancellation; this remains accurate
    // at the certified 10 GeV endpoint even when rrea::RreaKineticWide aliases double.
    rrea::RreaKineticWide const t_lo = inv_gamma_sq / (rrea::RreaKineticWide(1.0) + beta);  // mu = +1
    rrea::RreaKineticWide const t_hi = rrea::RreaKineticWide(1.0) + beta;                   // mu = -1
    auto const primitive = [c, b](rrea::RreaKineticWide t) {
        return -c / (rrea::RreaKineticWide(3.0) * t * t * t)
            - (rrea::RreaKineticWide(2.0) + b * c) / (rrea::RreaKineticWide(2.0) * t * t)
            - (rrea::RreaKineticWide(2.0) * b - rrea::RreaKineticWide(1.0)) / t
            - b * std::log(t);
    };

    rrea::RreaKineticWide const f_lo = primitive(t_lo);
    rrea::RreaKineticWide const f_hi = primitive(t_hi);
    rrea::RreaKineticWide const total = f_hi - f_lo;
    if (!(total > rrea::RreaKineticWide(0.0)) || !std::isfinite(total)) {
        rrea::KineticFail(
            "Sauter-Gavrila photoelectron inverse CDF has a non-positive/non-finite normalization");
    }
    rrea::RreaKineticWide const target = (rrea::RreaKineticWide(1.0) - u) * f_hi + u * f_lo;
    rrea::RreaKineticWide lo = t_lo;
    rrea::RreaKineticWide hi = t_hi;
    for (int iter = 0; iter < 128; ++iter) {
        rrea::RreaKineticWide const mid = rrea::RreaKineticWide(0.5) * (lo + hi);
        if (mid == lo || mid == hi) {
            break;
        }
        if (primitive(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    rrea::RreaKineticWide const t = rrea::RreaKineticWide(0.5) * (lo + hi);
    rrea::RreaKineticWide const mu = rrea::kinetic::max(-rrea::RreaKineticWide(1.0), rrea::kinetic::min(rrea::RreaKineticWide(1.0), (rrea::RreaKineticWide(1.0) - t) / beta));
    return static_cast<amrex::Real>(mu);
}

// ===== Pair-production final state =====
// This is a BANK REPLAY, not a distribution sampler: draw_invariant indexes
// the stored Geant4 final-state records and draw_pair_polar picks between
// adjacent energy blocks, so comparing its output to the bank validates the
// LOADER, not a sampled distribution.  The lepton polar/azimuth draws are
// deliberately unused (counter-based RNG: discarding them perturbs nothing);
// energy closure against the record is a hard abort at build time.
RREA_HOST_DEVICE inline RreaPairFinalState RreaInteractionTablesView::SamplePairFinalState(
    amrex::Real photon_energy_eV,
    RreaPhotonChannel channel,
    amrex::Real draw_invariant,
    amrex::Real draw_pair_polar,
    amrex::Real draw_pair_azimuth,
    amrex::Real draw_lepton_polar,
    amrex::Real draw_lepton_azimuth) const
{
    RequireSchema6SamplerEnergy(
        "SamplePairFinalState", "photon", m_photon_certified_range, photon_energy_eV);
    RreaPairFinalState state;
    state.channel = channel;

    bool const triplet = channel == RreaPhotonChannel::PairTriplet;
    bool const nuclear = channel == RreaPhotonChannel::PairNuclear;
    if (!triplet && !nuclear) {
        rrea::KineticFail("SamplePairFinalState called for a non-pair photon channel");
    }

    if (!RreaPairChannelHasPhaseSpace(photon_energy_eV, channel)) {
        return state;
    }

    if (m_pair_empirical_blocks.empty()) {
        rrea::KineticFail("schema-6 correlated pair final-state samples are not loaded");
    }
    // Blocks are stored under the split PairNuclear/PairTriplet channels.
    RreaPhotonChannel const block_channel = triplet
        ? RreaPhotonChannel::PairTriplet
        : RreaPhotonChannel::PairNuclear;
    // Nearest channel-specific brackets, without a temporary allocation.
    RreaPairEmpiricalBlockView const *first=nullptr,*last=nullptr,*lower=nullptr,*upper=nullptr;
    for (auto const& block : m_pair_empirical_blocks) {
        if (block.channel != block_channel) continue;
        if (!first) first=&block;
        last=&block;
        if (block.primary_energy_eV < photon_energy_eV) lower=&block;
        else if (!upper) upper=&block;
    }
    if (!first) rrea::KineticFail("schema-6 pair channel has no training blocks");
    RreaPairEmpiricalBlockView const* selected=first;
    if (photon_energy_eV >= last->primary_energy_eV) selected=last;
    else if (photon_energy_eV > first->primary_energy_eV) {
        if (upper->primary_energy_eV == photon_energy_eV) selected=upper;
        else {
            amrex::Real const t=(std::log(photon_energy_eV)-std::log(lower->primary_energy_eV))
                /(std::log(upper->primary_energy_eV)-std::log(lower->primary_energy_eV));
            selected=kinetic_tables::clamp_unit(draw_pair_polar)<t ? upper:lower;
        }
    }
    if (selected->training_records.empty()) {
        rrea::KineticFail("schema-6 selected pair final-state block is empty");
    }
    std::size_t const training_size =
        static_cast<std::size_t>(selected->training_records.size());
    std::size_t const record_index = rrea::kinetic::min(
        training_size - 1U,
        static_cast<std::size_t>(
            kinetic_tables::clamp_unit(draw_invariant)
            * static_cast<amrex::Real>(training_size)));
    auto const& record = selected->training_records[record_index];
    (void)draw_lepton_polar; (void)draw_lepton_azimuth;
    return BuildRreaPairEmpiricalFinalState(
        record, photon_energy_eV, channel, draw_pair_azimuth);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::PositronCollisionLoss(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return Require(RreaTable::PositronCollisionStopping).Interpolate(energy_eV)
        * density_ratio;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::PositronSoftRadiativeDrag(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return Require(RreaTable::PositronSoftRadiativeStopping).Interpolate(energy_eV)
        * density_ratio;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::PositronBremsstrahlungMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return kinetic_tables::mean_free_path_from_inverse_length(
        Require(RreaTable::PositronTrackedBremsInverseLength).Interpolate(energy_eV)
            * density_ratio);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::PositronHardBhabhaMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    amrex::Real const inverse_m_stp =
        Require(RreaTable::PositronHardBhabhaInverseLength).Interpolate(energy_eV);
    if (m_hard_moller_secondary_threshold_eV > amrex::Real(0.0)
        && energy_eV <= m_hard_moller_secondary_threshold_eV) {
        return amrex::Real(1.0e99);
    }
    return kinetic_tables::mean_free_path_from_inverse_length(
        inverse_m_stp * density_ratio);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::SamplePositronBhabhaSecondaryEnergy(
    amrex::Real primary_energy_eV,
    amrex::Real cdf_u) const
{
    RequireSchema6SamplerEnergy(
        "SamplePositronBhabhaSecondaryEnergy",
        "positron",
        m_charged_certified_range,
        primary_energy_eV);
    auto const* it = &m_conditional_cdfs[2];
    if (it->empty()) {
        rrea::KineticFail("positron_hard_bhabha_secondary_cdf_air is not loaded");
    }
    return kinetic_tables::sample_conditional_cdf(*it, primary_energy_eV, cdf_u, true);
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::PositronAnnihilationMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return kinetic_tables::mean_free_path_from_inverse_length(
        Require(RreaTable::PositronAnnihilationInverseLength).Interpolate(energy_eV)
            * density_ratio);
}

RREA_HOST_DEVICE inline RreaPositronAnnihilationFinalState RreaInteractionTablesView::SamplePositronAnnihilationFinalState(
    amrex::Real positron_kinetic_energy_eV,
    amrex::Real draw_energy,
    amrex::Real draw_azimuth) const
{
    RequireSchema6SamplerEnergy(
        "SamplePositronAnnihilationFinalState",
        "positron",
        m_charged_certified_range,
        positron_kinetic_energy_eV,
        /*allow_exact_zero=*/true);
    RreaPositronAnnihilationFinalState state;
    positron_kinetic_energy_eV = rrea::kinetic::max(amrex::Real(0.0), positron_kinetic_energy_eV);
    amrex::Real const gamma = amrex::Real(1.0) + positron_kinetic_energy_eV / kinetic_tables::kElectronRestEnergyEv;
    amrex::Real const total_photon_energy_eV =
        amrex::Real(2.0) * kinetic_tables::kElectronRestEnergyEv + positron_kinetic_energy_eV;
    if (gamma <= amrex::Real(1.0 + 1.0e-10)) {
        state.photon1_energy_eV = kinetic_tables::kElectronRestEnergyEv;
        state.photon2_energy_eV = kinetic_tables::kElectronRestEnergyEv;
        state.photon1_cos_theta = amrex::Real(2.0) * kinetic_tables::clamp_unit(draw_energy) - amrex::Real(1.0);
        state.photon1_azimuth_u = kinetic_tables::wrap_unit_interval(draw_azimuth);
        state.valid = true;
        state.at_rest = true;
        return state;
    }
    amrex::Real const beta = std::sqrt(rrea::kinetic::max(
        amrex::Real(0.0),
        amrex::Real(1.0) - amrex::Real(1.0) / (gamma * gamma)));
    amrex::Real const epsilon = kinetic_tables::sample_annihilation_photon_energy_fraction(gamma, draw_energy);
    state.photon1_energy_eV = epsilon * total_photon_energy_eV;
    state.photon2_energy_eV = rrea::kinetic::max(amrex::Real(0.0), total_photon_energy_eV - state.photon1_energy_eV);
    if (state.photon1_energy_eV <= amrex::Real(0.0) || state.photon2_energy_eV <= amrex::Real(0.0)) {
        return state;
    }
    amrex::Real const denom = epsilon * gamma * beta;
    if (denom <= amrex::Real(0.0)) {
        return state;
    }
    state.photon1_cos_theta = rrea::kinetic::max(
        amrex::Real(-1.0),
        rrea::kinetic::min(amrex::Real(1.0), (epsilon * (gamma + amrex::Real(1.0)) - amrex::Real(1.0)) / denom));
    state.photon1_azimuth_u = kinetic_tables::wrap_unit_interval(draw_azimuth);
    state.valid = true;
    state.at_rest = false;
    return state;
}

RREA_HOST_DEVICE inline amrex::Real RreaInteractionTablesView::PositronElasticTransportMeanFreePath(
    amrex::Real energy_eV,
    amrex::Real density_ratio) const
{
    if (density_ratio <= amrex::Real(0.0)) {
        rrea::KineticFail("density_ratio must be positive");
    }
    return kinetic_tables::mean_free_path_from_inverse_length(
        PositronTransportCoefficients(energy_eV, density_ratio)
            .elastic_transport_inverse_m);
}
} // rrea
#endif
