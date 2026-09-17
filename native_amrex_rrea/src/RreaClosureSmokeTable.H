#ifndef RREA_CLOSURE_SMOKE_TABLE_H
#define RREA_CLOSURE_SMOKE_TABLE_H

// Shared test-only helper: writes a synthetic electron-closure table whose
// rows satisfy the loader's derivation identities, so the fluid
// and advance smokes exercise the real Load() path without duplicating the
// writer.  Header-only, test TUs only; production bundles come from
// scripts/generate_rrea_fluid_closure_bundle.py.

#include "rrea/RreaConstants.H"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rrea_closure_smoke {

struct Row {
    double en_td;
    double w_1e5_cm_s;
    double eta_1e39_cm5;
    double eta2_1e19_cm2 = 0.0;
    double longitudinal_diffusion_ref_m2_per_s = 0.03;
    double transverse_diffusion_ref_m2_per_s = 0.05;
};

struct Derived {
    double k0 = 0.0;
    double eta_cm5 = 0.0;
    double k3 = 0.0;
    double nu3 = 0.0;
    double eta2_cm2 = 0.0;
    double k2 = 0.0;
    double nu2 = 0.0;
    double longitudinal_diffusion_ref = 0.0;
    double transverse_diffusion_ref = 0.0;
};

inline Derived derive(Row const& row)
{
    Derived out;
    double const w_m_s = row.w_1e5_cm_s * 1.0e3;
    out.k0 = w_m_s
        / (row.en_td * rrea::townsend_v_m2 * rrea::n_loschmidt_m3);
    out.eta_cm5 = row.eta_1e39_cm5 * 1.0e-39;
    out.k3 = (w_m_s * 100.0) * out.eta_cm5;
    double const n0_cm3 = rrea::n_loschmidt_m3 * 1.0e-6;
    out.nu3 = out.k3 * n0_cm3 * n0_cm3;
    out.eta2_cm2 = row.eta2_1e19_cm2 * 1.0e-19;
    out.k2 = (w_m_s * 100.0) * out.eta2_cm2;
    out.nu2 = out.k2 * n0_cm3;
    out.longitudinal_diffusion_ref =
        row.longitudinal_diffusion_ref_m2_per_s;
    out.transverse_diffusion_ref =
        row.transverse_diffusion_ref_m2_per_s;
    return out;
}

// A row whose derived pair is exactly (k0_target/..., nu3_target/...) up to
// round-off: invert the derivation identities for (w, eta) at this node.
inline Row row_for_targets(double en_td, double k0_target, double nu3_target)
{
    double const w_m_s =
        k0_target * en_td * rrea::townsend_v_m2 * rrea::n_loschmidt_m3;
    double const n0_cm3 = rrea::n_loschmidt_m3 * 1.0e-6;
    double const k3 = nu3_target / (n0_cm3 * n0_cm3);
    double const eta_cm5 = k3 / (w_m_s * 100.0);
    return Row{en_td, w_m_s / 1.0e3, eta_cm5 * 1.0e39};
}

inline std::string csv_text(
    std::vector<Row> const& rows,
    double two_body_value = 0.0,
    double k0_scale = 1.0)
{
    std::ostringstream csv;
    csv << "en_td,flux_drift_velocity_m_per_s,"
           "reduced_flux_mobility_ref_m2_per_v_s,"
           "longitudinal_diffusion_ref_m2_per_s,"
           "transverse_diffusion_ref_m2_per_s,eta_over_n2_cm5,"
           "three_body_rate_cm6_per_s,three_body_frequency_ref_s,"
           "eta_over_n_cm2,two_body_rate_cm3_per_s,"
           "two_body_frequency_ref_s,relative_uncertainty_mobility,"
           "relative_uncertainty_three_body,relative_uncertainty_two_body,"
           "source_row\n";
    csv << std::setprecision(17);
    for (std::size_t index = 0; index < rows.size(); ++index) {
        auto const derived = derive(rows[index]);
        csv << rows[index].en_td << ','
            << rows[index].w_1e5_cm_s * 1.0e3 << ','
            << derived.k0 * k0_scale << ','
            << derived.longitudinal_diffusion_ref << ','
            << derived.transverse_diffusion_ref << ','
            << derived.eta_cm5 << ','
            << derived.k3 << ','
            << derived.nu3 << ','
            << derived.eta2_cm2 << ',' << derived.k2 << ','
            << derived.nu2 + two_body_value
            << ",0.015,0.1,0.2,smoke_r" << index << '\n';
    }
    return csv.str();
}

inline std::string write_table(
    std::filesystem::path const& dir,
    std::string const& csv)
{
    std::filesystem::create_directories(dir);
    auto const path = dir / "electron_closure.csv";
    std::ofstream out(path, std::ios::binary);
    out << csv;
    return path.string();
}

}  // namespace rrea_closure_smoke

#endif
