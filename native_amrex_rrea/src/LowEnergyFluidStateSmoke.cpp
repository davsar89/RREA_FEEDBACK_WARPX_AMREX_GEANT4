#include "rrea/LowEnergyFluidState.H"
#include "rrea/RreaConstants.H"
#include "rrea/RreaInteractionTables.H"
#include "rrea/RreaSmokeRequire.H"
#include "RreaClosureSmokeTable.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

struct TestDomain {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
};

TestDomain make_domain(
    int nr,
    int nz,
    double r_max_m,
    double z_max_m,
    bool periodic_z = false,
    int index_offset = 0)
{
    amrex::IntVect lo(index_offset);
    amrex::IntVect hi(index_offset);
    hi[0] = index_offset + nr - 1;
#if (AMREX_SPACEDIM >= 2)
    hi[1] = index_offset + nz - 1;
#endif
#if (AMREX_SPACEDIM >= 3)
    lo[2] = 0;
    hi[2] = 0;
#endif
    amrex::Box domain(lo, hi);
    amrex::RealBox real_box;
    real_box.setLo(0, 0.0);
    real_box.setHi(0, r_max_m);
#if (AMREX_SPACEDIM >= 2)
    real_box.setLo(1, 0.0);
    real_box.setHi(1, z_max_m);
#endif
#if (AMREX_SPACEDIM >= 3)
    real_box.setLo(2, 0.0);
    real_box.setHi(2, 1.0);
#endif
    amrex::Vector<int> periodic(AMREX_SPACEDIM, 0);
#if (AMREX_SPACEDIM >= 2)
    if (periodic_z) {
        periodic[1] = 1;
    }
#else
    amrex::ignore_unused(periodic_z);
#endif
    amrex::Geometry geom(domain, &real_box, 0, periodic.data());
    amrex::BoxArray ba(domain);
    ba.maxSize(8);
    amrex::DistributionMapping dm(ba);
    return {geom, ba, dm};
}

// Cell-centred E onto the Yee faces the carrier transport reads: interior
// faces average the bracketing cells, the axis face is 0, physical outer
// faces clamp to the last cell.  Production commits the Maxwell faces
// directly; this is the smoke's stand-in for that field.
void cell_to_face_field(
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>& face_e,
    amrex::MultiFab const& er_cell,
    amrex::MultiFab const& ez_cell,
    amrex::Geometry const& geom)
{
    auto const domain = geom.Domain();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto const& cell = dir == 0 ? er_cell : ez_cell;
        amrex::MultiFab src(cell.boxArray(), cell.DistributionMap(), 1, 1);
        src.setVal(0.0);
        amrex::MultiFab::Copy(src, cell, 0, 0, 1, 0);
        src.FillBoundary(geom.periodicity());
        bool const periodic = geom.isPeriodic(dir);
        for (amrex::MFIter mfi(*face_e[dir]); mfi.isValid(); ++mfi) {
            auto const s = src.const_array(mfi);
            auto const out = face_e[dir]->array(mfi);
            amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
                amrex::IntVect const here(AMREX_D_DECL(i, j, k));
                amrex::IntVect below = here;
                below[dir] -= 1;
                if (!periodic && here[dir] <= domain.smallEnd(dir)) {
                    out(here) = dir == 0 ? amrex::Real(0.0) : s(here);
                } else if (!periodic && here[dir] > domain.bigEnd(dir)) {
                    out(here) = s(below);
                } else {
                    out(here) = amrex::Real(0.5) * (s(below) + s(here));
                }
            });
        }
        face_e[dir]->OverrideSync(geom.periodicity());
    }
}

void advance_carriers(
    rrea::LowEnergyFluidState& fluid,
    double dt,
    amrex::Geometry const& geom,
    amrex::MultiFab const& efield_x,
    amrex::MultiFab const& efield_z,
    rrea::LowEnergyFluidConfig const& config,
    amrex::MultiFab const* density_ratio = nullptr,
    bool collect_center_diagnostics = true)
{
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::runtime_error("carrier drift requires finite, non-negative dt");
    }
    fluid.ComputeFluidChargeDensity();
    if (dt == 0.0 || (!config.ion_drift_enable && !config.electron_drift_enable)) {
        return;
    }
    fluid.PrepareExplicitCarrierTransport(geom, config, density_ratio);

    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_e;
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> face_flux;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::BoxArray face_ba = fluid.PositiveIonDensity().boxArray();
        face_ba.surroundingNodes(dir);
        face_e[dir] = std::make_unique<amrex::MultiFab>(
            face_ba, fluid.PositiveIonDensity().DistributionMap(), 1, 0);
        face_flux[dir] = std::make_unique<amrex::MultiFab>(
            face_ba, fluid.PositiveIonDensity().DistributionMap(), 3, 0);
    }
    cell_to_face_field(face_e, efield_x, efield_z, geom);
    fluid.BuildExplicitFaceTransport(
        dt,
        geom,
        amrex::GetArrOfConstPtrs(face_e),
        config,
        amrex::GetArrOfPtrs(face_flux));
    fluid.ApplyExplicitCarrierTransport(
        dt,
        geom,
        amrex::GetArrOfPtrs(face_flux),
        config,
        density_ratio,
        collect_center_diagnostics);
}

double cell_volume_rz(amrex::Geometry const& geom, int i, int j)
{
    amrex::ignore_unused(j);
    auto const dx = geom.CellSizeArray();
    double const r_lo = geom.ProbLo(0)
        + static_cast<double>(i - geom.Domain().smallEnd(0)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
    return rrea::rz_shell_volume(r_lo, r_lo + dx[0], dx[1]);
#else
    return rrea::rz_shell_volume(r_lo, r_lo + dx[0], 1.0);
#endif
}

double center_z(
    amrex::MultiFab const& density,
    amrex::Geometry const& geom)
{
    double count = 0.0;
    double moment = 0.0;
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const domain = geom.Domain();
    for (amrex::MFIter mfi(density, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const arr = density.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
#if (AMREX_SPACEDIM >= 2)
            double const z = plo[1]
                + (static_cast<double>(j - domain.smallEnd(1)) + 0.5) * dx[1];
#else
            amrex::ignore_unused(j);
            double const z = 0.0;
#endif
            double const w = std::max(static_cast<double>(arr(i, j, k)), 0.0)
                * cell_volume_rz(geom, i, j);
            count += w;
            moment += w * z;
        });
    }
    amrex::ParallelDescriptor::ReduceRealSum(count);
    amrex::ParallelDescriptor::ReduceRealSum(moment);
    if (count <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return moment / count;
}

double total_count(
    amrex::MultiFab const& density,
    amrex::Geometry const& geom)
{
    double count = 0.0;
    for (amrex::MFIter mfi(density, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const arr = density.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            count += std::max(static_cast<double>(arr(i, j, k)), 0.0)
                * cell_volume_rz(geom, i, j);
        });
    }
    amrex::ParallelDescriptor::ReduceRealSum(count);
    return count;
}

double first_value(amrex::MultiFab const& field)
{
    for (amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const arr = field.const_array(mfi);
#if (AMREX_SPACEDIM >= 3)
        int const k = bx.smallEnd(2);
#else
        int const k = 0;
#endif
        return static_cast<double>(
            arr(bx.smallEnd(0), bx.smallEnd(1), k));
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void fill_gaussian(
    amrex::MultiFab& density,
    amrex::Geometry const& geom,
    double z0,
    double sigma_z)
{
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const domain = geom.Domain();
    for (amrex::MFIter mfi(density, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const arr = density.array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
#if (AMREX_SPACEDIM >= 2)
            double const z = plo[1]
                + (static_cast<double>(j - domain.smallEnd(1)) + 0.5) * dx[1];
#else
            amrex::ignore_unused(j);
            double const z = 0.0;
#endif
            double const dz = (z - z0) / sigma_z;
            arr(i, j, k) = static_cast<amrex::Real>(std::exp(-0.5 * dz * dz));
        });
    }
}

using rrea::smoke::require;
using rrea::smoke::require_throws;

void run_smoke()
{
    auto domain = make_domain(24, 128, 12.0, 128.0);
    rrea::LowEnergyFluidState fluid(domain.ba, domain.dm, 1);
    amrex::MultiFab er(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab ez(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab density_ratio(domain.ba, domain.dm, 1, 0);
    er.setVal(0.0);
    ez.setVal(1000.0);
    density_ratio.setVal(0.25);

    rrea::LowEnergyFluidConfig config;
    config.ion_drift_enable = true;
    config.ion_mobility_model = "reduced_stp_density_scaled";
    config.positive_ion_reduced_mobility_stp_m2_per_vs = 1.4e-4;
    config.negative_ion_reduced_mobility_stp_m2_per_vs = 1.9e-4;
    config.ion_mobility_density_ratio_floor = 0.05;
    config.ion_drift_cfl = 0.5;
    config.transport_density_ratio = 0.25;
    config.attachment_frequency_s = 0.0;
    config.ion_ion_recombination_coefficient_m3_per_s = 0.0;

    auto mobility_domain = make_domain(4, 4, 4.0, 4.0);
    rrea::LowEnergyFluidState mobility_fluid(
        mobility_domain.ba, mobility_domain.dm, 1);
    amrex::MultiFab mobility_density(
        mobility_domain.ba, mobility_domain.dm, 1, 0);
    mobility_density.setVal(0.25);
    mobility_fluid.PositiveIonDensity().setVal(1.0);
    mobility_fluid.ComputeConductivity(config, &mobility_density);
    double const inferred_mu_pos =
        first_value(mobility_fluid.Conductivity()) / rrea::qe;
    require(std::abs(inferred_mu_pos - 5.6e-4) < 1.0e-12,
            "density_ratio=0.25 did not give 4x positive reduced mobility");
    // Pins the min_tau_m reduced-CSV column the eps_QE diagnostic consumes:
    // tau_M = eps0/sigma, filled in the same kernel as the conductivity.
    require(std::abs(first_value(mobility_fluid.MaxwellTime())
                     * first_value(mobility_fluid.Conductivity())
                     - rrea::eps0) < 1.0e-24,
            "MaxwellTime() != eps0/Conductivity()");

    // Attachment and positive/negative recombination are charge-conserving
    // reactions.  Deliberately use unequal explicit source channels so the
    // check cannot pass merely because the total fluid charge is zero.
    {
        rrea::LowEnergyFluidState chemistry(
            mobility_domain.ba, mobility_domain.dm, 1);
        chemistry.LowElectronDensity().setVal(7.0);
        chemistry.PositiveIonDensity().setVal(11.0);
        chemistry.NegativeIonDensity().setVal(3.0);
        chemistry.LowElectronSourceRate().setVal(5.0);
        chemistry.PositiveIonSourceRate().setVal(9.0);
        chemistry.DirectNegativeIonSourceRate().setVal(2.0);
        // Direct carrier edits require a one-time conservative-charge
        // reconstruction; production initializes/restores rho_fluid before
        // chemistry and thereafter advances it from exact sources/currents.
        chemistry.ComputeFluidChargeDensity();
        rrea::LowEnergyFluidConfig chemistry_config;
        chemistry_config.attachment_frequency_s = 3.0;
        chemistry_config.ion_ion_recombination_coefficient_m3_per_s = 0.01;
        require(
            std::abs(chemistry.MaxReactionFrequency(chemistry_config) - 3.14)
                < 1.0e-14,
            "reaction-frequency bound omitted a chemistry channel");
        chemistry.UpdateFromIonizationSource(0.1, chemistry_config);
        require(
            chemistry.LastReactionChargeConservationResidual()
                <= 512.0 * std::numeric_limits<amrex::Real>::epsilon(),
            "charge-conserving fluid reactions changed local charge");
        require(
            std::abs(first_value(chemistry.FluidChargeDensity()) / rrea::qe - 1.2)
                < 1.0e-12,
            "fluid chemistry charge does not equal the explicit source charge");
    }

    // Two half reactions retain the outer-step source rate until the second
    // call and reproduce the exact constant-source attachment solution.
    {
        rrea::LowEnergyFluidState split(
            mobility_domain.ba, mobility_domain.dm, 1);
        split.LowElectronDensity().setVal(4.0);
        split.LowElectronSourceRate().setVal(5.0);
        split.ComputeFluidChargeDensity();
        rrea::LowEnergyFluidConfig split_config;
        split_config.attachment_frequency_s = 3.0;
        split.UpdateFromIonizationSource(
            0.05, split_config, nullptr, nullptr, false);
        require(std::abs(first_value(split.LowElectronSourceRate()) - 5.0)
                    < 1.0e-15,
                "first half reaction consumed the preserved source rate");
        split.UpdateFromIonizationSource(0.05, split_config);
        double const expected = 4.0 * std::exp(-0.3)
            + 5.0 * (-std::expm1(-0.3) / 3.0);
        require(std::abs(first_value(split.LowElectronDensity()) - expected)
                    < 1.0e-14,
                "split source/attachment integral changed the exact solution");
        require(std::abs(first_value(split.FluidChargeDensity()) / rrea::qe + 4.5)
                    < 1.0e-14,
                "split source integral changed conservative fluid charge");
        require(first_value(split.LowElectronSourceRate()) == 0.0,
                "final half reaction did not consume its source rate");
    }

    // Exact ion-ion pair loss (dn+/dt = dn-/dt = -alpha n+ n-) against
    // high-precision RK4 over the four regimes the closure guide names:
    // equal densities, strongly unequal, weak reaction, stiff reaction.
    {
        struct PairRegime { double np0; double nn0; double alpha; double dt; };
        PairRegime const regimes[] = {
            {10.0, 10.0, 0.01, 0.1},
            {1000.0, 1.0, 0.01, 0.1},
            {10.0, 7.0, 1.0e-6, 0.1},
            {100.0, 90.0, 1.0, 1.0},
        };
        for (auto const& regime : regimes) {
            rrea::LowEnergyFluidState pair(
                mobility_domain.ba, mobility_domain.dm, 1);
            pair.PositiveIonDensity().setVal(regime.np0);
            pair.NegativeIonDensity().setVal(regime.nn0);
            pair.ComputeFluidChargeDensity();
            rrea::LowEnergyFluidConfig pair_config;
            pair_config.attachment_frequency_s = 0.0;
            pair_config.ion_ion_recombination_coefficient_m3_per_s = regime.alpha;
            pair.UpdateFromIonizationSource(regime.dt, pair_config);
            double np_ref = regime.np0;
            double nn_ref = regime.nn0;
            int const n_sub = 100000;
            double const h = regime.dt / n_sub;
            for (int s = 0; s < n_sub; ++s) {
                auto rate = [&](double a, double b) {
                    return -regime.alpha * a * b;
                };
                double const k1 = rate(np_ref, nn_ref);
                double const k2 = rate(np_ref + 0.5 * h * k1, nn_ref + 0.5 * h * k1);
                double const k3 = rate(np_ref + 0.5 * h * k2, nn_ref + 0.5 * h * k2);
                double const k4 = rate(np_ref + h * k3, nn_ref + h * k3);
                double const dn = h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                np_ref += dn;
                nn_ref += dn;
            }
            require(
                std::abs(first_value(pair.PositiveIonDensity()) - np_ref)
                    <= 1.0e-9 * regime.np0,
                "exact pair-loss n+ deviates from high-precision integration");
            require(
                std::abs(first_value(pair.NegativeIonDensity()) - nn_ref)
                    <= 1.0e-9 * regime.np0,
                "exact pair-loss n- deviates from high-precision integration");
        }
    }

    // Detachment: exact linear transfer n_neg -> n_e, charge conserving.
    {
        rrea::LowEnergyFluidState det(
            mobility_domain.ba, mobility_domain.dm, 1);
        det.LowElectronDensity().setVal(2.0);
        det.PositiveIonDensity().setVal(9.0);
        det.NegativeIonDensity().setVal(5.0);
        det.ComputeFluidChargeDensity();
        rrea::LowEnergyFluidConfig det_config;
        det_config.attachment_frequency_s = 0.0;
        det_config.detachment_frequency_s = 4.0;
        det.UpdateFromIonizationSource(0.3, det_config);
        double const survived = 5.0 * std::exp(-4.0 * 0.3);
        require(
            std::abs(first_value(det.NegativeIonDensity()) - survived)
                < 1.0e-12 * 5.0,
            "detachment does not follow the exact linear decay");
        require(
            std::abs(first_value(det.LowElectronDensity())
                     - (2.0 + 5.0 - survived)) < 1.0e-12 * 5.0,
            "detached electrons were not returned to the electron fluid");
    }

    // Conductivity and persisted drift charge must use exactly the same
    // enabled species.  Disabled carriers may intentionally carry zero or
    // otherwise-unused mobility settings.
    {
        rrea::LowEnergyFluidState enabled_fluid(
            mobility_domain.ba, mobility_domain.dm, 1);
        enabled_fluid.LowElectronDensity().setVal(2.0);
        enabled_fluid.PositiveIonDensity().setVal(3.0);
        enabled_fluid.NegativeIonDensity().setVal(5.0);

        rrea::LowEnergyFluidConfig enabled_config;
        enabled_config.electron_drift_enable = true;
        enabled_config.ion_drift_enable = false;
        enabled_config.electron_mobility_model = "absolute";
        enabled_config.electron_mobility_m2_per_vs = 0.25;
        enabled_config.ion_mobility_model = "unused-invalid-model";
        enabled_config.positive_ion_mobility_m2_per_vs = 0.0;
        enabled_config.negative_ion_mobility_m2_per_vs = 0.0;
        enabled_fluid.ComputeConductivity(enabled_config);
        require(
            std::abs(first_value(enabled_fluid.Conductivity())
                    - rrea::qe * 0.25 * 2.0)
                < 1.0e-30,
            "disabled ions contributed to conductivity");

        enabled_config.electron_drift_enable = false;
        enabled_config.ion_drift_enable = true;
        enabled_config.ion_mobility_model = "absolute";
        enabled_config.positive_ion_mobility_m2_per_vs = 0.2;
        enabled_config.negative_ion_mobility_m2_per_vs = 0.3;
        enabled_config.electron_mobility_model = "unused-invalid-model";
        enabled_config.electron_mobility_m2_per_vs = 0.0;
        enabled_fluid.ComputeConductivity(enabled_config);
        require(
            std::abs(first_value(enabled_fluid.Conductivity())
                    - rrea::qe * (0.2 * 3.0 + 0.3 * 5.0))
                < 1.0e-30,
            "disabled electrons contributed to conductivity");

        enabled_config.electron_drift_enable = true;
        enabled_config.electron_mobility_model = "absolute";
        enabled_config.electron_mobility_m2_per_vs = 0.25;
        enabled_fluid.ComputeConductivity(enabled_config);
        require(
            std::abs(first_value(enabled_fluid.Conductivity())
                    - rrea::qe * (0.25 * 2.0 + 0.2 * 3.0 + 0.3 * 5.0))
                < 1.0e-30,
            "simultaneously enabled electron/ion conductivity is inconsistent");

        enabled_config.electron_drift_enable = false;
        enabled_config.ion_drift_enable = false;
        enabled_config.ion_mobility_model = "unused-invalid-model";
        enabled_config.positive_ion_mobility_m2_per_vs = 0.0;
        enabled_config.negative_ion_mobility_m2_per_vs = 0.0;
        enabled_fluid.ComputeConductivity(enabled_config);
        require(
            first_value(enabled_fluid.Conductivity()) == 0.0,
            "conductivity was nonzero with every carrier disabled");

        enabled_config.electron_drift_enable = true;
        enabled_config.electron_mobility_model = "absolute";
        enabled_config.electron_mobility_m2_per_vs = 0.0;
        require_throws(
            [&]() { enabled_fluid.ComputeConductivity(enabled_config); },
            "enabled zero-mobility electron did not fail closed");

        enabled_config.electron_drift_enable = false;
        enabled_config.ion_drift_enable = true;
        enabled_config.positive_ion_mobility_m2_per_vs = 0.0;
        enabled_config.negative_ion_mobility_m2_per_vs = 0.3;
        require_throws(
            [&]() { enabled_fluid.ComputeConductivity(enabled_config); },
            "enabled zero-mobility positive ion did not fail closed");
    }

    // Analytic drift-enable matrix: the same enabled-species contract used by
    // conductivity must control the persisted carrier advection.  With a
    // uniform axial field and an interior Gaussian, the finite-volume first
    // moment translates by sign(q)*mu*Ez*dt while conserving particle count.
    auto require_drift_close = [](double actual, double expected, char const* message) {
        require(std::abs(actual - expected) < 1.0e-8, message);
    };
    {
        double const dt = 0.2;
        double const electron_mu = 2.0e-3;

        rrea::LowEnergyFluidState electron_only(domain.ba, domain.dm, 1);
        fill_gaussian(electron_only.LowElectronDensity(), domain.geom, 72.0, 4.0);
        fill_gaussian(electron_only.PositiveIonDensity(), domain.geom, 72.0, 4.0);
        double const electron_z0 = center_z(
            electron_only.LowElectronDensity(), domain.geom);
        double const positive_z0 = center_z(
            electron_only.PositiveIonDensity(), domain.geom);
        double const electron_total0 = total_count(
            electron_only.LowElectronDensity(), domain.geom);
        double const positive_total0 = total_count(
            electron_only.PositiveIonDensity(), domain.geom);
        rrea::LowEnergyFluidConfig electron_only_config;
        electron_only_config.electron_drift_enable = true;
        electron_only_config.ion_drift_enable = false;
        electron_only_config.electron_mobility_model = "absolute";
        electron_only_config.electron_mobility_m2_per_vs = electron_mu;
        electron_only_config.ion_mobility_model = "unused-invalid-model";
        electron_only_config.positive_ion_mobility_m2_per_vs = 0.0;
        electron_only_config.negative_ion_mobility_m2_per_vs = 0.0;
        advance_carriers(
            electron_only,
            dt,
            domain.geom,
            er,
            ez,
            electron_only_config,
            &density_ratio);
        require_drift_close(
            center_z(electron_only.LowElectronDensity(), domain.geom) - electron_z0,
            -electron_mu * 1000.0 * dt,
            "electron-only drift did not match -mu_e*Ez*dt");
        require_drift_close(
            center_z(electron_only.PositiveIonDensity(), domain.geom),
            positive_z0,
            "disabled positive ions moved in electron-only drift");
        require(
            std::abs(total_count(electron_only.LowElectronDensity(), domain.geom)
                    - electron_total0) / electron_total0 < 1.0e-12,
            "electron-only drift did not conserve electron count");
        require(
            std::abs(total_count(electron_only.PositiveIonDensity(), domain.geom)
                    - positive_total0) / positive_total0 < 1.0e-12,
            "electron-only drift changed disabled positive-ion count");

        // The speed diagnostic must divide a face flux by its upwind donor
        // density, not by the density of a nearly empty recipient cell.  A
        // negative electron velocity carries density 1 across this sharp front
        // into density 1e-30; the physical speed remains exactly 1 m/s.
        {
            auto front_domain = make_domain(2, 4, 2.0, 4.0);
            rrea::LowEnergyFluidState front_fluid(
                front_domain.ba, front_domain.dm, 1);
            amrex::MultiFab front_er(front_domain.ba, front_domain.dm, 1, 0);
            amrex::MultiFab front_ez(front_domain.ba, front_domain.dm, 1, 0);
            front_er.setVal(0.0);
            front_ez.setVal(4.0);
            front_fluid.LowElectronDensity().setVal(1.0e-30);
            int const front_j = front_domain.geom.Domain().smallEnd(1) + 2;
            for (amrex::MFIter mfi(
                     front_fluid.LowElectronDensity(),
                     amrex::TilingIfNotGPU());
                 mfi.isValid();
                 ++mfi) {
                auto const bx = mfi.validbox();
                auto const density = front_fluid.LowElectronDensity().array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
                    if (j >= front_j) {
                        density(i, j, k) = 1.0;
                    }
                });
            }
            rrea::LowEnergyFluidConfig front_config;
            front_config.electron_drift_enable = true;
            front_config.ion_drift_enable = false;
            front_config.electron_mobility_model = "absolute";
            front_config.electron_mobility_m2_per_vs = 0.25;
            front_config.ion_drift_cfl = 0.5;
            constexpr double front_dt = 0.1;
            advance_carriers(
                front_fluid,
                front_dt,
                front_domain.geom,
                front_er,
                front_ez,
                front_config);
            auto const& front_drift =
                front_fluid.LastCarrierDriftDiagnostics();
            require(
                std::abs(front_drift.max_drift_speed_m_per_s - 1.0) < 1.0e-12,
                "sharp-front drift diagnostic divided by recipient density");
            require(
                std::abs(front_drift.cfl_dt_s - 0.5) < 1.0e-12
                    && std::abs(front_drift.max_outgoing_cfl - front_dt)
                        < 1.0e-12
                    && front_drift.substeps == 1,
                "sharp-front drift diagnostic changed the physical CFL");
        }

        // A deliberately super-CFL cell is limited conservatively rather
        // than aborting the whole Maxwell step. The test computes the exact
        // one-cell donor bound independently: unlimited CFL 1 -> scale 0.5.
        {
            auto limited_domain = make_domain(2, 4, 2.0, 4.0, true);
            rrea::LowEnergyFluidState limited(
                limited_domain.ba, limited_domain.dm, 1);
            amrex::MultiFab limited_er(
                limited_domain.ba, limited_domain.dm, 1, 0);
            amrex::MultiFab limited_ez(
                limited_domain.ba, limited_domain.dm, 1, 0);
            limited_er.setVal(0.0);
            limited_ez.setVal(4.0);
            limited.LowElectronDensity().setVal(0.0);
            int const donor_j = limited_domain.geom.Domain().smallEnd(1) + 2;
            for (amrex::MFIter mfi(limited.LowElectronDensity()); mfi.isValid(); ++mfi) {
                auto const bx = mfi.validbox();
                auto const density = limited.LowElectronDensity().array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
                    if (j == donor_j) {
                        density(i, j, k) = 1.0;
                    }
                });
            }
            double const count_before = total_count(
                limited.LowElectronDensity(), limited_domain.geom);
            rrea::LowEnergyFluidConfig limited_config;
            limited_config.electron_drift_enable = true;
            limited_config.ion_drift_enable = false;
            limited_config.electron_mobility_model = "absolute";
            limited_config.electron_mobility_m2_per_vs = 0.25;
            limited_config.ion_drift_cfl = 0.5;
            advance_carriers(
                limited, 1.0, limited_domain.geom,
                limited_er, limited_ez, limited_config);
            auto const& diag = limited.LastCarrierDriftDiagnostics();
            double const cfl_roundoff = 512.0
                * std::numeric_limits<double>::epsilon();
            require(
                std::abs(diag.unlimited_max_outgoing_cfl - 1.0) < 1.0e-12
                    && diag.min_outflow_scale > 0.0
                    && diag.min_outflow_scale < 1.0
                    && diag.max_outgoing_cfl
                        <= limited_config.ion_drift_cfl + cfl_roundoff
                    && diag.limited_outflow_fraction > 0.0
                    && diag.limited_electron_number_fraction > 0.0,
                "conservative face limiter did not enforce the donor bound");
            require(
                limited.LowElectronDensity().min(0, 0, false) >= -1.0e-14
                    && std::abs(total_count(
                        limited.LowElectronDensity(), limited_domain.geom)
                        - count_before) <= 1.0e-12 * count_before,
                "face limiter broke positivity or periodic carrier conservation");
            double max_charge_error = 0.0;
            for (amrex::MFIter mfi(limited.LowElectronDensity()); mfi.isValid(); ++mfi) {
                auto const bx = mfi.validbox();
                auto const electron = limited.LowElectronDensity().const_array(mfi);
                auto const charge = limited.FluidChargeDensity().const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
                    max_charge_error = std::max(
                        max_charge_error,
                        std::abs(static_cast<double>(charge(i, j, k))
                            + rrea::qe * static_cast<double>(electron(i, j, k))));
                });
            }
            amrex::ParallelDescriptor::ReduceRealMax(max_charge_error);
            require(
                max_charge_error <= 512.0
                    * std::numeric_limits<double>::epsilon() * rrea::qe,
                "limited electron current and conservative fluid charge diverged");
        }

        rrea::LowEnergyFluidState both_enabled(domain.ba, domain.dm, 1);
        fill_gaussian(both_enabled.LowElectronDensity(), domain.geom, 72.0, 4.0);
        fill_gaussian(both_enabled.PositiveIonDensity(), domain.geom, 72.0, 4.0);
        fill_gaussian(both_enabled.NegativeIonDensity(), domain.geom, 72.0, 4.0);
        double const both_e_z0 = center_z(
            both_enabled.LowElectronDensity(), domain.geom);
        double const both_p_z0 = center_z(
            both_enabled.PositiveIonDensity(), domain.geom);
        double const both_n_z0 = center_z(
            both_enabled.NegativeIonDensity(), domain.geom);
        double const both_e_total0 = total_count(
            both_enabled.LowElectronDensity(), domain.geom);
        double const both_p_total0 = total_count(
            both_enabled.PositiveIonDensity(), domain.geom);
        double const both_n_total0 = total_count(
            both_enabled.NegativeIonDensity(), domain.geom);
        rrea::LowEnergyFluidConfig both_config = config;
        both_config.electron_drift_enable = true;
        both_config.ion_drift_enable = true;
        both_config.electron_mobility_model = "absolute";
        both_config.electron_mobility_m2_per_vs = electron_mu;
        advance_carriers(
            both_enabled,
            dt,
            domain.geom,
            er,
            ez,
            both_config,
            &density_ratio);
        require_drift_close(
            center_z(both_enabled.LowElectronDensity(), domain.geom) - both_e_z0,
            -electron_mu * 1000.0 * dt,
            "both-enabled electron drift did not match -mu_e*Ez*dt");
        require_drift_close(
            center_z(both_enabled.PositiveIonDensity(), domain.geom) - both_p_z0,
            (1.4e-4 / 0.25) * 1000.0 * dt,
            "both-enabled positive-ion drift did not match +mu_pos*Ez*dt");
        require_drift_close(
            center_z(both_enabled.NegativeIonDensity(), domain.geom) - both_n_z0,
            -(1.9e-4 / 0.25) * 1000.0 * dt,
            "both-enabled negative-ion drift did not match -mu_neg*Ez*dt");
        require(
            std::abs(total_count(both_enabled.LowElectronDensity(), domain.geom)
                    - both_e_total0) / both_e_total0 < 1.0e-12
                && std::abs(total_count(both_enabled.PositiveIonDensity(), domain.geom)
                    - both_p_total0) / both_p_total0 < 1.0e-12
                && std::abs(total_count(both_enabled.NegativeIonDensity(), domain.geom)
                    - both_n_total0) / both_n_total0 < 1.0e-12,
            "both-enabled drift did not conserve every carrier count");

        rrea::LowEnergyFluidState all_disabled(domain.ba, domain.dm, 1);
        fill_gaussian(all_disabled.LowElectronDensity(), domain.geom, 72.0, 4.0);
        fill_gaussian(all_disabled.PositiveIonDensity(), domain.geom, 72.0, 4.0);
        fill_gaussian(all_disabled.NegativeIonDensity(), domain.geom, 72.0, 4.0);
        double const disabled_e_z0 = center_z(
            all_disabled.LowElectronDensity(), domain.geom);
        double const disabled_p_z0 = center_z(
            all_disabled.PositiveIonDensity(), domain.geom);
        double const disabled_n_z0 = center_z(
            all_disabled.NegativeIonDensity(), domain.geom);
        rrea::LowEnergyFluidConfig disabled_config;
        disabled_config.electron_drift_enable = false;
        disabled_config.ion_drift_enable = false;
        disabled_config.electron_mobility_model = "unused-invalid-model";
        disabled_config.ion_mobility_model = "unused-invalid-model";
        disabled_config.electron_mobility_m2_per_vs = 0.0;
        disabled_config.positive_ion_mobility_m2_per_vs = 0.0;
        disabled_config.negative_ion_mobility_m2_per_vs = 0.0;
        advance_carriers(
            all_disabled,
            dt,
            domain.geom,
            er,
            ez,
            disabled_config,
            &density_ratio);
        require_drift_close(
            center_z(all_disabled.LowElectronDensity(), domain.geom),
            disabled_e_z0,
            "all-disabled drift moved low-energy electrons");
        require_drift_close(
            center_z(all_disabled.PositiveIonDensity(), domain.geom),
            disabled_p_z0,
            "all-disabled drift moved positive ions");
        require_drift_close(
            center_z(all_disabled.NegativeIonDensity(), domain.geom),
            disabled_n_z0,
            "all-disabled drift moved negative ions");
        require(
            all_disabled.LastCarrierDriftDiagnostics().substeps == 0,
            "all-disabled drift performed an advection substep");
    }

    // Carriers crossing a periodic z seam must wrap, not vanish. Place the positive-ion
    // blob against the upper seam and drift it upward far enough that most of
    // it crosses; exact conservation catches an absorbing-seam
    // defect, and the wrapped mass must reappear in the lower quarter.
    {
        auto periodic_domain = make_domain(24, 128, 12.0, 128.0, true);
        rrea::LowEnergyFluidState periodic_fluid(
            periodic_domain.ba, periodic_domain.dm, 1);
        amrex::MultiFab er_p(periodic_domain.ba, periodic_domain.dm, 1, 0);
        amrex::MultiFab ez_p(periodic_domain.ba, periodic_domain.dm, 1, 0);
        amrex::MultiFab den_p(periodic_domain.ba, periodic_domain.dm, 1, 0);
        er_p.setVal(0.0);
        ez_p.setVal(1000.0);
        den_p.setVal(0.25);
        fill_gaussian(
            periodic_fluid.PositiveIonDensity(), periodic_domain.geom, 127.5, 0.15);
        double const wrap_total0 = total_count(
            periodic_fluid.PositiveIonDensity(), periodic_domain.geom);
        auto lower_quarter_count = [&]() {
            double count = 0.0;
            auto const& density = periodic_fluid.PositiveIonDensity();
            for (amrex::MFIter mfi(density, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const bx = mfi.validbox();
                auto const arr = density.const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
#if (AMREX_SPACEDIM >= 2)
                    if (j >= 32) {
                        return;
                    }
#endif
                    count += std::max(static_cast<double>(arr(i, j, k)), 0.0)
                        * cell_volume_rz(periodic_domain.geom, i, j);
                });
            }
            amrex::ParallelDescriptor::ReduceRealSum(count);
            return count;
        };
        double const wrap_low0 = lower_quarter_count();
        rrea::LowEnergyFluidConfig wrap_config = config;
        wrap_config.electron_drift_enable = false;
        wrap_config.ion_drift_enable = true;
        // +mu_pos/den * Ez * dt = 0.42 m upward.  The narrow blob occupies the
        // last one-metre cell, so its 0.42 outgoing CFL transfers a resolvable
        // fraction through the periodic seam without hidden subcycling.
        advance_carriers(
            periodic_fluid,
            0.75,
            periodic_domain.geom,
            er_p,
            ez_p,
            wrap_config,
            &den_p);
        double const wrap_total1 = total_count(
            periodic_fluid.PositiveIonDensity(), periodic_domain.geom);
        require(
            std::abs(wrap_total1 - wrap_total0) / wrap_total0 < 1.0e-12,
            "periodic-z drift did not conserve carriers across the seam");
        require(
            lower_quarter_count() - wrap_low0 > 0.3 * wrap_total0,
            "periodic-z drift did not wrap carriers across the seam");
    }

    fill_gaussian(fluid.PositiveIonDensity(), domain.geom, 72.0, 4.0);
    fill_gaussian(fluid.NegativeIonDensity(), domain.geom, 72.0, 4.0);
    double const pos_z0 = center_z(fluid.PositiveIonDensity(), domain.geom);
    double const neg_z0 = center_z(fluid.NegativeIonDensity(), domain.geom);
    double const pos_total0 = total_count(fluid.PositiveIonDensity(), domain.geom);
    double const neg_total0 = total_count(fluid.NegativeIonDensity(), domain.geom);

    advance_carriers(
        fluid,
        0.5,
        domain.geom,
        er,
        ez,
        config,
        &density_ratio);

    auto const& drift = fluid.LastCarrierDriftDiagnostics();
    require(drift.substeps > 0, "ion drift did not report substeps");
    double const pos_z1 = center_z(fluid.PositiveIonDensity(), domain.geom);
    double const neg_z1 = center_z(fluid.NegativeIonDensity(), domain.geom);
    require(pos_z1 > pos_z0, "positive ions did not drift with +Ez");
    require(neg_z1 < neg_z0, "negative ions did not drift opposite +Ez");
    require(std::abs(total_count(fluid.PositiveIonDensity(), domain.geom) - pos_total0)
                / std::max(pos_total0, 1.0) < 1.0e-8,
            "positive ion count changed in interior drift smoke");
    require(std::abs(total_count(fluid.NegativeIonDensity(), domain.geom) - neg_total0)
                / std::max(neg_total0, 1.0) < 1.0e-8,
            "negative ion count changed in interior drift smoke");

    // AMReX permits index domains whose small end is not zero.  Geometry
    // coordinates and RZ shell volumes must depend on the offset from the
    // domain small end, not on the raw integer index.
    {
        auto shifted = make_domain(8, 32, 4.0, 32.0, false, 7);
        rrea::LowEnergyFluidState shifted_fluid(shifted.ba, shifted.dm, 1);
        amrex::MultiFab shifted_er(shifted.ba, shifted.dm, 1, 0);
        amrex::MultiFab shifted_ez(shifted.ba, shifted.dm, 1, 0);
        amrex::MultiFab shifted_density(shifted.ba, shifted.dm, 1, 0);
        shifted_er.setVal(0.0);
        shifted_ez.setVal(1000.0);
        shifted_density.setVal(0.25);
        fill_gaussian(
            shifted_fluid.PositiveIonDensity(), shifted.geom, 16.0, 2.0);
        double const shifted_z0 = center_z(
            shifted_fluid.PositiveIonDensity(), shifted.geom);
        double const shifted_total0 = total_count(
            shifted_fluid.PositiveIonDensity(), shifted.geom);
        require(
            std::abs(shifted_z0 - 16.0) < 1.0e-10,
            "nonzero-index geometry mapped the carrier center incorrectly");
        advance_carriers(
            shifted_fluid,
            0.5,
            shifted.geom,
            shifted_er,
            shifted_ez,
            config,
            &shifted_density);
        double const shifted_z1 = center_z(
            shifted_fluid.PositiveIonDensity(), shifted.geom);
        require(
            std::abs(shifted_z1 - shifted_z0 - 0.28) < 1.0e-8,
            "nonzero-index geometry changed the physical drift distance");
        require(
            std::abs(total_count(
                         shifted_fluid.PositiveIonDensity(), shifted.geom)
                    - shifted_total0)
                    / shifted_total0
                < 1.0e-12,
            "nonzero-index geometry did not conserve carrier count");
    }

    // The constant attachment model applies attachment_frequency_s
    // everywhere: ne(dt) = ne0 * exp(-nu*dt), independent of the density
    // field it is handed.
    {
        auto att_domain = make_domain(4, 4, 4.0, 4.0);
        double const dt = 1.0e-7;
        rrea::LowEnergyFluidConfig const_config;
        const_config.attachment_model = "constant";
        const_config.attachment_frequency_s = 2.0e6;
        rrea::LowEnergyFluidState const_fluid(att_domain.ba, att_domain.dm, 1);
        amrex::MultiFab att_density(att_domain.ba, att_domain.dm, 1, 0);
        att_density.setVal(0.25);
        const_fluid.LowElectronDensity().setVal(1.0e12);
        const_fluid.UpdateFromIonizationSource(dt, const_config, &att_density);
        double const expected_const = 1.0e12 * std::exp(-2.0e6 * dt);
        require(std::abs(first_value(const_fluid.LowElectronDensity()) - expected_const)
                    / expected_const < 1.0e-12,
                "constant attachment model changed behavior");
    }

    auto domain_zero = make_domain(8, 16, 4.0, 16.0);
    rrea::LowEnergyFluidState zero(domain_zero.ba, domain_zero.dm, 1);
    amrex::MultiFab zerofield(domain_zero.ba, domain_zero.dm, 1, 0);
    amrex::MultiFab zerodensity(domain_zero.ba, domain_zero.dm, 1, 0);
    zerofield.setVal(0.0);
    zerodensity.setVal(0.25);
    fill_gaussian(zero.PositiveIonDensity(), domain_zero.geom, 8.0, 2.0);
    double const zero_z0 = center_z(zero.PositiveIonDensity(), domain_zero.geom);
    advance_carriers(
        zero,
        10.0,
        domain_zero.geom,
        zerofield,
        zerofield,
        config,
        &zerodensity);
    double const zero_z1 = center_z(zero.PositiveIonDensity(), domain_zero.geom);
    require(std::abs(zero_z1 - zero_z0) < 1.0e-12, "zero-field drift moved ions");
}

// ---- semantic electron-closure loader --------------------------------------
//
// Hermetic: writes its own synthetic table (a subset of real de Urquijo
// rows, regenerated through the same derivation identities the loader
// enforces), then exercises semantic validation, node reproduction,
// log-midpoint interpolation, and the hold-below/forbid-above range policy.

using rrea_closure_smoke::Row;
auto const& closure_smoke_derive = rrea_closure_smoke::derive;
auto const& closure_smoke_csv = rrea_closure_smoke::csv_text;

void run_electron_closure_smoke()
{
    namespace fs = std::filesystem;
    fs::path const dir = fs::temp_directory_path() / "rrea_closure_smoke";
    fs::remove_all(dir);
    auto write_table = [&dir](std::string const& csv) {
        return rrea_closure_smoke::write_table(dir, csv);
    };

    std::vector<Row> const rows = {
        {4.0, 12.4, 38.40, 0.0}, {9.0, 18.9, 12.20, 2.0},
        {16.0, 27.8, 6.81, 4.0}};
    std::string const table_path =
        write_table(closure_smoke_csv(rows, 0.0, 1.0));

    auto const closure =
        rrea::LowEnergyElectronClosure::Load(table_path);
    require(closure.Loaded()
                && closure.ModelId()
                    == "dry_air_swarm_hybrid_v3_flux_attachment_diffusion"
                && closure.MinEnTd() == 4.0 && closure.MaxEnTd() == 16.0
                && closure.InterpolationId()
                    == "piecewise_log_linear_positive_with_nonnegative_onset",
            "closure table metadata did not load faithfully");

    // Exact node reproduction, including the derived arithmetic itself.
    for (auto const& row : rows) {
        auto const expected = closure_smoke_derive(row);
        auto const got = closure.EvaluateHost(row.en_td);
        require(std::abs(got.k0_flux_ref_m2_per_vs - expected.k0)
                        <= 1.0e-15 * expected.k0
                    && std::abs(got.nu3_ref_per_s - expected.nu3)
                        <= 1.0e-15 * expected.nu3
                    && std::abs(got.nu2_ref_per_s - expected.nu2)
                        <= 1.0e-15 * std::max(expected.nu2, 1.0)
                    && std::abs(
                           got.longitudinal_diffusion_ref_m2_per_s
                           - expected.longitudinal_diffusion_ref)
                        <= 1.0e-15 * expected.longitudinal_diffusion_ref
                    && std::abs(
                           got.transverse_diffusion_ref_m2_per_s
                           - expected.transverse_diffusion_ref)
                        <= 1.0e-15 * expected.transverse_diffusion_ref,
                "closure node evaluation did not reproduce the table row");
    }
    // Log-midpoint of a log-linear interpolant is the geometric mean.
    {
        double const x_mid = std::sqrt(4.0 * 9.0);
        auto const got = closure.EvaluateHost(x_mid);
        auto const lo = closure_smoke_derive(rows[0]);
        auto const hi = closure_smoke_derive(rows[1]);
        require(std::abs(got.k0_flux_ref_m2_per_vs - std::sqrt(lo.k0 * hi.k0))
                        <= 1.0e-12 * std::sqrt(lo.k0 * hi.k0)
                    && std::abs(got.nu3_ref_per_s - std::sqrt(lo.nu3 * hi.nu3))
                        <= 1.0e-12 * std::sqrt(lo.nu3 * hi.nu3)
                    && std::abs(got.nu2_ref_per_s - 0.5 * hi.nu2)
                        <= 1.0e-12 * hi.nu2
                    && std::abs(
                           got.longitudinal_diffusion_ref_m2_per_s
                           - std::sqrt(
                               lo.longitudinal_diffusion_ref
                               * hi.longitudinal_diffusion_ref))
                        <= 1.0e-12 * std::sqrt(
                            lo.longitudinal_diffusion_ref
                            * hi.longitudinal_diffusion_ref)
                    && std::abs(
                           got.transverse_diffusion_ref_m2_per_s
                           - std::sqrt(
                               lo.transverse_diffusion_ref
                               * hi.transverse_diffusion_ref))
                        <= 1.0e-12 * std::sqrt(
                            lo.transverse_diffusion_ref
                            * hi.transverse_diffusion_ref),
                "log-midpoint interpolation is not the geometric mean");
    }
    // Range policy: hold below (counted by callers), forbid above, reject
    // non-finite.
    {
        rrea::RreaClosureRangeStatus status{};
        auto const held = closure.Evaluate(3.5, status);
        auto const node0 = closure_smoke_derive(rows[0]);
        require(status == rrea::RreaClosureRangeStatus::HeldBelow
                    && std::abs(held.k0_flux_ref_m2_per_vs - node0.k0)
                        <= 1.0e-15 * node0.k0,
                "below-range evaluation must hold the lowest node");
        (void)closure.Evaluate(16.5, status);
        require(status == rrea::RreaClosureRangeStatus::AboveRange,
                "above-range evaluation must report AboveRange");
        require_throws(
            [&]() { (void)closure.EvaluateHost(16.5); },
            "EvaluateHost accepted an above-range E/N");
        (void)closure.Evaluate(
            std::numeric_limits<double>::quiet_NaN(), status);
        require(status == rrea::RreaClosureRangeStatus::Invalid,
                "non-finite E/N must report Invalid");
    }
    // Semantic failures: a broken two-body chain and a broken K0 derivation.
    auto require_load_rejected = [&write_table](
                                     std::string const& csv,
                                     char const* message) {
        auto const rejected = write_table(csv);
        require_throws(
            [&rejected]() {
                (void)rrea::LowEnergyElectronClosure::Load(rejected);
            },
            message);
    };
    require_load_rejected(
        closure_smoke_csv(rows, 1.0, 1.0),
        "loader accepted a broken two-body derivation chain");
    require_load_rejected(
        closure_smoke_csv(rows, 0.0, 1.01),
        "loader accepted a row violating the derivation identity");
    fs::remove_all(dir);
}

// ---- closure cell-field builder --------------------------------------------
//
// The centering contract: coefficients come from the COMPLETE cell |E| and
// the actual positive neutral density.  Manufactured states pin the |E|
// composition (a 3-4-5 field equals a 0-5 field), the density invariance at
// fixed E/N (mu ~ 1/chi, nu = nu2*chi + nu3*chi^2), hold-below counting, and
// invalid-cell accounting the advance-level abort gate consumes.
void run_closure_builder_smoke()
{
    namespace fs = std::filesystem;
    fs::path const dir =
        fs::temp_directory_path() / "rrea_closure_builder_smoke";
    fs::remove_all(dir);
    std::vector<Row> const rows = {
        {4.0, 12.4, 38.40, 0.0}, {9.0, 18.9, 12.20, 2.0},
        {16.0, 27.8, 6.81, 4.0}};
    auto const table_path = rrea_closure_smoke::write_table(
        dir, closure_smoke_csv(rows, 0.0, 1.0));
    auto const closure =
        rrea::LowEnergyElectronClosure::Load(table_path);

    auto domain = make_domain(4, 4, 4.0, 4.0);
    amrex::MultiFab er(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab ez(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab chi(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab ne(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab source(domain.ba, domain.dm, 1, 0);
    amrex::MultiFab mu(domain.ba, domain.dm, 1, 1);
    amrex::MultiFab nu(domain.ba, domain.dm, 1, 1);
    amrex::MultiFab diffusion(domain.ba, domain.dm, 3, 1);
    ne.setVal(2.0e12);
    source.setVal(3.0e18);
    // Row j = 0: |E| = 5e4 via (3,4)e4 components at chi = 0.25.
    // Row j = 1: |E| = 5e4 via (0,5)e4 at chi = 0.25 (same coefficients).
    // Row j = 2: |E| = 1e5, chi = 0.5 -> SAME E/N; mu halves, nu quadruples.
    // Row j = 3: zero field (held below) at chi = 0.25.
    auto const domain_lo = domain.geom.Domain().smallEnd();
    for (amrex::MFIter mfi(er, false); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const er_arr = er.array(mfi);
        auto const ez_arr = ez.array(mfi);
        auto const chi_arr = chi.array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            amrex::ignore_unused(i);
            int const row = j - domain_lo[1];
            double er_value = 0.0;
            double ez_value = 0.0;
            double chi_value = 0.25;
            if (row == 0) {
                er_value = 3.0e4;
                ez_value = 4.0e4;
            } else if (row == 1) {
                ez_value = -5.0e4;
            } else if (row == 2) {
                er_value = 6.0e4;
                ez_value = 8.0e4;
                chi_value = 0.5;
            }
            er_arr(i, j, k) = er_value;
            ez_arr(i, j, k) = ez_value;
            chi_arr(i, j, k) = chi_value;
        });
    }
    rrea::LowEnergyClosureFieldDiagnostics diag;
    rrea::BuildElectronClosureCellFields(
        domain.geom, er, ez, &chi, 1.0, closure, &ne, &source,
        &mu, &nu, &diffusion, diag);
    // 4 columns x 1 held row; nothing above range or invalid.  Rows 0-2 sit
    // at E/N = 5e4 * 1e21 / (0.25 * N0) = 7.44 Td, inside [4, 16].
    require(diag.above_range_cells == 0 && diag.invalid_cells == 0
                && diag.held_below_cells == 4,
            "builder range accounting is wrong for the manufactured state");
    auto value_at = [&](amrex::MultiFab const& field, int row) {
        for (amrex::MFIter mfi(field, false); mfi.isValid(); ++mfi) {
            auto const bx = mfi.validbox();
            auto const arr = field.const_array(mfi);
            if (bx.smallEnd(1) <= domain_lo[1] + row
                && domain_lo[1] + row <= bx.bigEnd(1)) {
                return static_cast<double>(
                    arr(bx.smallEnd(0), domain_lo[1] + row, 0));
            }
        }
        throw std::runtime_error("row not found on this rank layout");
    };
    double const mu_row0 = value_at(mu, 0);
    double const nu_row0 = value_at(nu, 0);
    auto const closure_at_row0 = closure.EvaluateHost(
        5.0e4 * 1.0e21 / (0.25 * rrea::n_loschmidt_m3));
    require(std::abs(value_at(mu, 1) - mu_row0) <= 1.0e-15 * mu_row0
                && std::abs(value_at(nu, 1) - nu_row0) <= 1.0e-15 * nu_row0,
            "component composition changed the coefficients: |E| must be "
            "hypot(Er, Ez), not a single component");
    double const expected_nu_row2 =
        closure_at_row0.nu2_ref_per_s * 0.5
        + closure_at_row0.nu3_ref_per_s * 0.25;
    require(std::abs(value_at(mu, 2) - 0.5 * mu_row0) <= 1.0e-12 * mu_row0
                && std::abs(value_at(nu, 2) - expected_nu_row2)
                    <= 1.0e-12 * expected_nu_row2,
            "density scaling failed for mobility or two-/three-body attachment");
    auto diffusion_value_at = [&](int component, int row) {
        for (amrex::MFIter mfi(diffusion, false); mfi.isValid(); ++mfi) {
            auto const bx = mfi.validbox();
            auto const arr = diffusion.const_array(mfi);
            if (bx.smallEnd(1) <= domain_lo[1] + row
                && domain_lo[1] + row <= bx.bigEnd(1)) {
                return static_cast<double>(
                    arr(bx.smallEnd(0), domain_lo[1] + row, 0, component));
            }
        }
        throw std::runtime_error("diffusion row not found on this rank layout");
    };
    double const d_l =
        closure_at_row0.longitudinal_diffusion_ref_m2_per_s / 0.25;
    double const d_t =
        closure_at_row0.transverse_diffusion_ref_m2_per_s / 0.25;
    require(
        std::abs(diffusion_value_at(0, 0)
                 - (d_t + (d_l - d_t) * 0.36)) <= 1.0e-12 * d_t
            && std::abs(diffusion_value_at(1, 0)
                        - (d_l - d_t) * 0.48) <= 1.0e-12 * d_t
            && std::abs(diffusion_value_at(2, 0)
                        - (d_t + (d_l - d_t) * 0.64)) <= 1.0e-12 * d_t,
        "closure builder did not form the field-aligned diffusion tensor");
    require(
        std::abs(diffusion_value_at(0, 1) - d_t) <= 1.0e-12 * d_t
            && std::abs(diffusion_value_at(1, 1)) <= 1.0e-14 * d_t
            && std::abs(diffusion_value_at(2, 1) - d_l) <= 1.0e-12 * d_l,
        "axial field did not map D_T/D_L onto the RZ tensor");
    require(
        std::abs(diffusion_value_at(0, 2)
                 - 0.5 * diffusion_value_at(0, 0)) <= 1.0e-12 * d_t
            && std::abs(diffusion_value_at(1, 2)
                        - 0.5 * diffusion_value_at(1, 0)) <= 1.0e-12 * d_t
            && std::abs(diffusion_value_at(2, 2)
                        - 0.5 * diffusion_value_at(2, 0)) <= 1.0e-12 * d_l,
        "diffusion did not scale as 1/density at fixed E/N");
    double const held_d_t =
        closure_smoke_derive(rows[0]).transverse_diffusion_ref / 0.25;
    require(
        std::abs(diffusion_value_at(0, 3) - held_d_t)
                <= 1.0e-12 * held_d_t
            && std::abs(diffusion_value_at(1, 3)) <= 1.0e-14 * held_d_t
            && std::abs(diffusion_value_at(2, 3) - held_d_t)
                <= 1.0e-12 * held_d_t,
        "zero field must have isotropic transverse diffusion");
    double const k0_lowest = closure_smoke_derive(rows[0]).k0;
    double const mu_held = k0_lowest / 0.25;
    require(std::abs(value_at(mu, 3) - k0_lowest / 0.25)
                <= 1.0e-12 * (k0_lowest / 0.25),
            "held-below cells must carry the lowest node's coefficients");
    double const electron_held_fraction =
        diag.held_below_electron_count / diag.electron_count;
    double const source_held_fraction =
        diag.held_below_electron_source_rate_per_s
        / diag.electron_source_rate_per_s;
    double const conductivity_held_fraction =
        diag.held_below_electron_conductivity_proxy_m2_per_v_s
        / diag.electron_conductivity_proxy_m2_per_v_s;
    double const expected_conductivity_fraction =
        mu_held / (2.5 * mu_row0 + mu_held);
    require(std::abs(electron_held_fraction - 0.25) <= 1.0e-14
                && std::abs(source_held_fraction - 0.25) <= 1.0e-14
                && std::abs(
                       conductivity_held_fraction
                       - expected_conductivity_fraction)
                    <= 1.0e-14,
            "builder did not distinguish active held-below plasma from quiet "
            "held cells with the declared weights");
    // Above-range diagnostics retain the offending active cell details for
    // the collective advance-level abort message.
    for (amrex::MFIter mfi(ez, false); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        if (bx.contains(domain_lo)) {
            ez.array(mfi)(domain_lo[0], domain_lo[1], 0) = 2.0e5;
        }
    }
    rrea::LowEnergyClosureFieldDiagnostics above;
    rrea::BuildElectronClosureCellFields(
        domain.geom, er, ez, &chi, 1.0, closure, &ne, &source,
        &mu, &nu, &diffusion, above);
    require(
        above.above_range_cells == 1
            && above.max_above_range_i == domain_lo[0]
            && above.max_above_range_j == domain_lo[1]
            && above.max_above_range_electron_density_m3 > 0.0
            && above.max_above_range_source_rate_m3_s > 0.0
            && above.max_above_range_conductivity_proxy_per_m_v_s > 0.0,
        "above-range active-cell details were not retained");
    // A non-finite field component is counted and the cell skipped.
    {
        for (amrex::MFIter mfi(ez, false); mfi.isValid(); ++mfi) {
            auto const bx = mfi.validbox();
            auto const ez_arr = ez.array(mfi);
            if (bx.contains(domain_lo)) {
                ez_arr(domain_lo[0], domain_lo[1], 0) =
                    std::numeric_limits<amrex::Real>::quiet_NaN();
            }
        }
        rrea::LowEnergyClosureFieldDiagnostics bad;
        rrea::BuildElectronClosureCellFields(
            domain.geom, er, ez, &chi, 1.0, closure, &ne, &source,
            &mu, &nu, &diffusion, bad);
        require(bad.invalid_cells == 1,
                "a non-finite field cell must be counted as invalid");
    }
    fs::remove_all(dir);
}

// ---- table-attachment chemistry --------------------------------------------
//
// The en_table path must integrate the exact exponential with expm1 so tiny
// nu*dt keeps full precision, and fail closed on binding mismatches.
void run_table_attachment_smoke()
{
    auto att_domain = make_domain(4, 4, 4.0, 4.0);
    amrex::MultiFab density(att_domain.ba, att_domain.dm, 1, 0);
    density.setVal(0.25);
    amrex::MultiFab nu_field(att_domain.ba, att_domain.dm, 1, 1);

    rrea::LowEnergyFluidConfig table_config;
    table_config.attachment_model = "en_table_attachment_v1";

    double const dt = 1.0e-7;
    double const source_rate = 5.0e18;
    for (double nu : {2.5e7, 1.0e-5}) {
        // 1e-5 * 1e-7 = 1e-12: the regime where (1 - exp(-x))/nu loses ~4
        // digits to cancellation while -expm1(-x)/nu stays exact.
        rrea::LowEnergyFluidState fluid(att_domain.ba, att_domain.dm, 1);
        fluid.LowElectronDensity().setVal(1.0e12);
        fluid.LowElectronSourceRate().setVal(source_rate);
        fluid.ComputeFluidChargeDensity();
        nu_field.setVal(nu);
        fluid.UpdateFromIonizationSource(dt, table_config, &density, &nu_field);
        double const x = nu * dt;
        double const expected = 1.0e12 * std::exp(-x)
            + source_rate * (-std::expm1(-x) / nu);
        double const got = first_value(fluid.LowElectronDensity());
        require(std::abs(got - expected) <= 1.0e-12 * expected,
                "table attachment did not match the exact expm1 form");
        if (nu == 1.0e-5) {
            // Distinguishes expm1 from the naive (1-exp)/nu spelling: at
            // x = 1e-12 the naive source integral is off by ~1e-4 relative.
            double const series_factor = dt * (1.0 - 0.5 * x + x * x / 6.0);
            require(std::abs((-std::expm1(-x) / nu) - series_factor)
                        <= 1.0e-10 * series_factor,
                    "expm1 source factor drifted from its series value");
            require(std::abs(got - (1.0e12 * std::exp(-x)
                                    + source_rate * series_factor))
                        <= 1.0e-10 * expected,
                    "tiny-nu*dt attachment lost precision (naive 1-exp?)");
        }
    }

    // Fail-closed bindings: a table model without its field, and a field
    // handed to a non-table model.
    {
        rrea::LowEnergyFluidState fluid(att_domain.ba, att_domain.dm, 1);
        require_throws(
            [&]() {
                fluid.UpdateFromIonizationSource(dt, table_config, &density);
            },
            "en_table attachment without its frequency field did not throw");
        rrea::LowEnergyFluidConfig const_config;
        const_config.attachment_model = "constant";
        require_throws(
            [&]() {
                fluid.UpdateFromIonizationSource(
                    dt, const_config, &density, &nu_field);
            },
            "constant attachment accepted a bound frequency field");
    }
}

}  // namespace

int main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    int result = 0;
    try {
        run_smoke();
        run_electron_closure_smoke();
        run_closure_builder_smoke();
        run_table_attachment_smoke();
        amrex::Print() << "rrea_low_energy_fluid_smoke passed\n";
    } catch (std::exception const& exc) {
        result = 1;
        amrex::AllPrint() << "rrea_low_energy_fluid_smoke failed: "
                          << exc.what() << "\n";
    }
    amrex::Finalize();
    return result;
}
