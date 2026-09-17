// Analytic pins for the field-based reduced-CSV observables
// (rrea/RreaReducedObservables.H) -- the numbers every analysis quotes.
// Each case is exact by construction on a 4x4 RZ box with dr = dz = 1:
// cell volumes are pi(2i+1), a z-split at j = 2 halves the volume exactly,
// and every expected value below is either closed-form or the documented
// histogram convention (geometric/linear bin center, clamped into the
// realized span) evaluated with the same constants the header uses.

#include "rrea/RreaConstants.H"
#include "rrea/RreaReducedObservables.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <cmath>
#include <iostream>
#include <string>

namespace {

constexpr double kPi = rrea::pi;

struct TestDomain {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
};

TestDomain make_domain(int nr, int nz)
{
    amrex::IntVect lo(0);
    amrex::IntVect hi(0);
    hi[0] = nr - 1;
#if (AMREX_SPACEDIM >= 2)
    hi[1] = nz - 1;
#endif
    amrex::Box domain(lo, hi);
    amrex::RealBox real_box;
    real_box.setLo(0, 0.0);
    real_box.setHi(0, static_cast<double>(nr));
#if (AMREX_SPACEDIM >= 2)
    real_box.setLo(1, 0.0);
    real_box.setHi(1, static_cast<double>(nz));
#endif
    TestDomain out;
    out.geom.define(
        domain, &real_box, 0,
        amrex::Vector<int>(AMREX_SPACEDIM, 0).data());
    out.ba = amrex::BoxArray(domain);
    out.dm = amrex::DistributionMapping(out.ba);
    return out;
}

void check(bool ok, std::string const& what)
{
    if (!ok) {
        std::cerr << "FAIL " << what << "\n";
        amrex::Abort("rrea_reduced_observables_smoke: " + what);
    }
    std::cout << "PASS " << what << "\n";
}

bool close(double a, double b, double rel = 1.0e-12)
{
    double const scale = std::max({std::abs(a), std::abs(b), 1.0e-300});
    return std::abs(a - b) <= rel * scale;
}

template <typename F>
void fill(amrex::MultiFab& mf, F&& f)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto arr = mf.array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
            arr(i, j, k) = static_cast<amrex::Real>(f(i, j));
        });
    }
}

void run()
{
    auto const dom = make_domain(4, 4);
    amrex::MultiFab ez(dom.ba, dom.dm, 1, 0);
    amrex::MultiFab er(dom.ba, dom.dm, 1, 0);
    amrex::MultiFab aux(dom.ba, dom.dm, 1, 0);
    amrex::MultiFab aux2(dom.ba, dom.dm, 1, 0);
    // Total region volume: pi r_max^2 H = pi * 4^2 * 4 (radial ring volumes
    // pi(2i+1) summed over i = 0..3 give 16 pi per unit z, times nz = 4).
    double const v_total = 64.0 * kPi;

    // --- acceleration_field_metrics_rz: uniform field ------------------
    fill(ez, [](int, int) { return -80.0; });
    {
        auto const m = rrea::acceleration_field_metrics_rz(
            ez, dom.geom, 100.0, 0.0, 4.0, 1.0e9);
        check(close(m.mean_fraction, 0.8) && close(m.min_fraction, 0.8)
                  && close(m.max_fraction, 0.8) && close(m.p10_fraction, 0.8),
              "uniform field: every fraction statistic is |Ez|/E0 = 0.8");
        check(m.fraction_below_half == 0.0 && close(m.volume_m3, v_total)
                  && close(m.mean_abs_e_v_per_m, 80.0),
              "uniform field: below-half share 0, exact volume and mean |E|");
    }

    // --- acceleration_field_metrics_rz: equal-volume z split -----------
    fill(ez, [](int, int j) { return j < 2 ? 20.0 : 100.0; });
    {
        auto const m = rrea::acceleration_field_metrics_rz(
            ez, dom.geom, 100.0, 0.0, 4.0, 1.0e9);
        check(close(m.mean_fraction, 0.6) && close(m.fraction_below_half, 0.5)
                  && close(m.min_fraction, 0.2) && close(m.max_fraction, 1.0),
              "split field: volume-weighted mean 0.6, half below one half");
        // p10 target (10% of volume) falls in the first of 512 linear bins:
        // reported value is that bin's center, min + 0.5*span/512.
        check(close(m.p10_fraction, 0.2 + 0.5 * 0.8 / 512.0),
              "split field: p10 is the first linear-histogram bin center");
    }

    // --- weighted_field_fraction_rz ------------------------------------
    fill(ez, [](int i, int j) {
        if (i == 0 && j == 0) { return 40.0; }
        if (i == 2 && j == 1) { return 100.0; }
        return 0.0;
    });
    fill(aux, [](int i, int j) {
        if (i == 0 && j == 0) { return 3.0; }
        if (i == 2 && j == 1) { return 1.0; }
        return 0.0;
    });
    {
        auto const m = rrea::weighted_field_fraction_rz(
            ez, aux, dom.geom, 100.0, 0.0, 4.0, 1.0e9);
        // Volumes pi(2i+1): w*V = 3pi and 5pi -> mean = (3pi*0.4 + 5pi*1.0)/8pi.
        check(close(m.fraction, (3.0 * 0.4 + 5.0 * 1.0) / 8.0)
                  && close(m.weight_integral, 8.0 * kPi),
              "weighted fraction: exact two-cell weighted mean 0.775");
    }

    // --- en_td_region_metrics_rz: uniform 10 Td ------------------------
    double const chi = 0.25;
    auto e_of_td = [chi](double td) {
        return td * rrea::townsend_v_m2 * chi * rrea::n_loschmidt_m3;
    };
    er.setVal(0.0);
    fill(ez, [&](int, int) { return e_of_td(10.0); });
    aux.setVal(1.0);   // ne_low
    aux2.setVal(1.0);  // ionization source
    {
        auto const m = rrea::en_td_region_metrics_rz(
            er, ez, nullptr, static_cast<amrex::Real>(chi), aux, aux2,
            dom.geom, 0.0, 4.0, 1.0e9);
        check(close(m.min_en_td, 10.0) && close(m.max_en_td, 10.0),
              "uniform E/N: realized span is exactly 10 Td");
        check(close(m.weighted_mean_en_td[0], 10.0)
                  && close(m.weighted_mean_en_td[1], 10.0)
                  && close(m.weighted_mean_en_td[2], 10.0),
              "uniform E/N: all weighted means are exactly 10 Td");
        // Degenerate span: the clamp pins every percentile to the span itself.
        bool percentiles_ok = true;
        for (int channel = 0; channel < 3; ++channel) {
            for (int q = 0; q < 3; ++q) {
                percentiles_ok = percentiles_ok
                    && m.p10_p50_p90[channel][q] == m.min_en_td;
            }
        }
        check(percentiles_ok,
              "uniform E/N: all nine percentiles clamp to the realized value");
        check(m.vol_frac_below_3td == 0.0 && m.vol_frac_above_30td == 0.0
                  && m.ne_frac_below_3td == 0.0,
              "uniform 10 Td: certified-range fractions are exactly zero");
    }

    // --- en_td_region_metrics_rz: 2 Td / 50 Td split -------------------
    fill(ez, [&](int, int j) { return e_of_td(j < 2 ? 2.0 : 50.0); });
    fill(aux, [](int, int j) { return j < 2 ? 1.0 : 0.0; });   // ne in cold half
    fill(aux2, [](int, int j) { return j < 2 ? 0.0 : 1.0; });  // source in hot half
    {
        auto const m = rrea::en_td_region_metrics_rz(
            er, ez, nullptr, static_cast<amrex::Real>(chi), aux, aux2,
            dom.geom, 0.0, 4.0, 1.0e9);
        check(close(m.vol_frac_below_3td, 0.5)
                  && close(m.vol_frac_above_30td, 0.5)
                  && close(m.ne_frac_below_3td, 1.0),
              "split E/N: exact half-volume 3/30 Td fractions, ne all cold");
        check(close(m.weighted_mean_en_td[0], 26.0)
                  && close(m.weighted_mean_en_td[1], 2.0)
                  && close(m.weighted_mean_en_td[2], 50.0),
              "split E/N: volume/ne/source weighted means separate exposure");
        // The 2 Td log-bin's geometric center lies below the realized
        // minimum, so the cold-channel percentiles clamp to min exactly.
        check(m.p10_p50_p90[1][0] == m.min_en_td
                  && m.p10_p50_p90[1][1] == m.min_en_td
                  && m.p10_p50_p90[1][2] == m.min_en_td,
              "split E/N: ne-channel percentiles clamp to the 2 Td minimum");
        // Hot source channel: geometric center of the 50 Td log bin,
        // computed with the header's own binning convention.
        double const log_lo = std::log(0.03);
        double const inv_w = 160.0 / (std::log(300.0) - log_lo);
        int const bin = static_cast<int>(
            (std::log(50.0 * chi * rrea::n_loschmidt_m3 * rrea::townsend_v_m2
                      * 1.0e21 / (chi * rrea::n_loschmidt_m3)) - log_lo)
            * inv_w);
        double const center = std::exp(log_lo + (bin + 0.5) / inv_w);
        check(close(m.p10_p50_p90[2][1], center),
              "split E/N: source-channel median is the 50 Td log-bin center");
    }

    // --- ohmic_discharge_metrics_rz: single conduction ring ------------
    fill(ez, [](int i, int j) { return (i == 2 && j == 1) ? -50.0 : 0.0; });
    fill(aux, [](int i, int j) { return (i == 2 && j == 1) ? 2.0 : 0.0; });
    {
        auto const m = rrea::ohmic_discharge_metrics_rz(
            ez, aux, dom.geom, 0.0, 4.0, 1.0e9);
        double const ring_v = 5.0 * kPi;  // pi(2i+1), i = 2
        double const mean_abs = 2.0 * 50.0 * ring_v / v_total;
        check(close(m.mean_sigma_abs_e_A_per_m2, mean_abs)
                  && close(m.mean_sigma_ez_A_per_m2, -mean_abs),
              "ohmic ring: exact mean sigma|E| and signed sigma Ez");
        check(close(m.fluid_current_rms_r_m, 2.5)
                  && m.fluid_current_rms_z_m == 0.0,
              "ohmic ring: rms_r is the ring radius, rms_z vanishes");
        check(close(m.predicted_abs_e_decay_v_m_s, -mean_abs / rrea::eps0)
                  && close(m.current_proxy_A, -mean_abs * v_total / 4.0),
              "ohmic ring: decay rate -<sigma|E|>/eps0 and current proxy");
    }
}

}  // namespace

int main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    run();
    std::cout << "rrea_reduced_observables_smoke: all pins passed\n";
    amrex::Finalize();
    return 0;
}
