#include "rrea/LowEnergyFluidState.H"

#include "rrea/RreaConstants.H"
#include "rrea/RreaFieldHandoff.H"
#include "rrea/RreaInteractionTables.H"

#include <AMReX_Array4.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MFIter.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rrea {

namespace {

// Reduction layout shared by the closure kernel's success/invalid exits.
using ClosureOps = amrex::ReduceOps<
    amrex::ReduceOpMin, amrex::ReduceOpMax, amrex::ReduceOpSum,
    amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpMax,
    amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
    amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
    amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax,
    amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax>;
using ClosureData = amrex::ReduceData<
    double, double, long, long, long, double,
    double, double, double, double, double, double,
    double, double, double, double, double, double>;
using ClosureTuple = typename ClosureData::Type;

RREA_HOST_DEVICE ClosureTuple closure_tuple(LowEnergyClosureFieldDiagnostics const& diagnostics)
{
    return {
        diagnostics.min_en_td,
        diagnostics.max_en_td,
        diagnostics.held_below_cells,
        diagnostics.above_range_cells,
        diagnostics.invalid_cells,
        diagnostics.max_above_range_en_td,
        diagnostics.electron_count,
        diagnostics.held_below_electron_count,
        diagnostics.electron_source_rate_per_s,
        diagnostics.held_below_electron_source_rate_per_s,
        diagnostics.electron_conductivity_proxy_m2_per_v_s,
        diagnostics.held_below_electron_conductivity_proxy_m2_per_v_s,
        diagnostics.max_electron_density_m3,
        diagnostics.max_held_below_electron_density_m3,
        diagnostics.max_electron_source_rate_m3_s,
        diagnostics.max_held_below_electron_source_rate_m3_s,
        diagnostics.max_electron_conductivity_proxy_per_m_v_s,
        diagnostics.max_held_below_electron_conductivity_proxy_per_m_v_s};
}

// One RZ cell's volume and face areas (radial faces 2 pi r dz, axial faces
// pi (r_hi^2 - r_lo^2)).
struct FluidRzCell {
    double volume, axial_area, radial_area_lo, radial_area_hi;
};

struct FluidRzMetric {
    double r_lo, dr, dz;
    int i_lo;
    RREA_HOST_DEVICE double CellVolume(int i) const noexcept
    {
        double const inner = r_lo + (i - i_lo) * dr;
        return rz_shell_volume(inner, inner + dr, dz);
    }
    RREA_HOST_DEVICE FluidRzCell Cell(int i) const noexcept
    {
        double const inner = r_lo + static_cast<double>(i - i_lo) * dr;
        double const outer = inner + dr;
        return {CellVolume(i), pi * std::max(outer * outer - inner * inner, 0.0),
                2.0 * pi * inner * dz, 2.0 * pi * outer * dz};
    }
};

// Number leaving cell (i, j) per unit time through its four faces.
template <class FluxArray>
RREA_HOST_DEVICE double outgoing_number_rate(
    FluxArray const& flux_r, FluxArray const& flux_z,
    int i, int j, int k, int species, FluidRzCell const& cell) noexcept
{
    return std::max(static_cast<double>(flux_r(i + 1, j, k, species)), 0.0)
            * cell.radial_area_hi
        + std::max(-static_cast<double>(flux_r(i, j, k, species)), 0.0)
            * cell.radial_area_lo
        + (std::max(static_cast<double>(flux_z(i, j + 1, k, species)), 0.0)
           + std::max(-static_cast<double>(flux_z(i, j, k, species)), 0.0))
            * cell.axial_area;
}

FluidRzMetric fluid_metric(amrex::Geometry const& geom)
{
    return {geom.ProbLo(0), geom.CellSize(0), geom.CellSize(1),
            geom.Domain().smallEnd(0)};
}

// The detailed above-range report is an exceptional host read of one cell,
// not a full-grid copy. This also works with a non-managed AMReX arena.
double host_cell(amrex::Array4<amrex::Real const> const& a, int i, int j, int k)
{
    amrex::Real value;
    auto const* ptr = &a(i, j, k);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, ptr, ptr + 1, &value);
    return static_cast<double>(value);
}

} // namespace

void LowEnergyFluidState::ValidateCarrierTransportConfig(
    LowEnergyFluidConfig const& config,
    bool require_drift_cfl)
{
    // Ions carry the density-scaled arm; electrons carry the table arm.
    // Both keep "absolute", which is the smokes' analytic control.
    auto valid_model = [](std::string const& model, bool electron) {
        return model == "absolute"
            || (electron ? model == "en_table_flux_v1"
                         : model == "reduced_stp_density_scaled");
    };
    auto positive = [](double value) {
        return std::isfinite(value) && value > 0.0;
    };
    auto check = [&](char const* species,
                     std::string const& model,
                     double absolute,
                     double reduced,
                     bool electron = false) {
        if (!valid_model(model, electron)) {
            throw std::runtime_error(
                std::string("enabled ") + species
                + " carrier has invalid mobility model: " + model);
        }
        if (model == "en_table_flux_v1") {
            // The scalar constants are unused: the caller-evaluated cell
            // mobility field carries the closure and is validated where it
            // is bound.
            return;
        }
        double const selected = model == "absolute" ? absolute : reduced;
        if (!positive(selected)) {
            throw std::runtime_error(
                std::string("enabled ") + species
                + " carrier requires a positive finite mobility");
        }
    };
    bool const drift_ions = config.ion_drift_enable;
    bool const drift_electrons = config.electron_drift_enable;
    if (!drift_ions && !drift_electrons) {
        return;
    }
    if (drift_ions) {
        check(
            "positive-ion",
            config.ion_mobility_model,
            config.positive_ion_mobility_m2_per_vs,
            config.positive_ion_reduced_mobility_stp_m2_per_vs);
        check(
            "negative-ion",
            config.ion_mobility_model,
            config.negative_ion_mobility_m2_per_vs,
            config.negative_ion_reduced_mobility_stp_m2_per_vs);
    }
    if (drift_electrons) {
        check(
            "low-energy-electron",
            config.electron_mobility_model,
            config.electron_mobility_m2_per_vs,
            /*reduced=*/0.0,
            /*electron=*/true);
    }
    bool const uses_reduced = drift_ions
        && config.ion_mobility_model == "reduced_stp_density_scaled";
    if (uses_reduced
        && (!positive(config.ion_mobility_density_ratio_floor)
            || !positive(config.transport_density_ratio))) {
        throw std::runtime_error(
            "density-scaled carrier mobility requires positive density ratios");
    }
    if (require_drift_cfl
        && (!std::isfinite(config.ion_drift_cfl)
        || config.ion_drift_cfl <= 0.0
        || config.ion_drift_cfl > 1.0)) {
        throw std::runtime_error("rrea.ion_drift_cfl must be finite and in (0, 1]");
    }
}

void LowEnergyFluidState::PrepareExplicitCarrierTransport(
    amrex::Geometry const& geom,
    LowEnergyFluidConfig const& config,
    amrex::MultiFab const* density_ratio,
    double source_dt_s)
{
    BL_PROFILE("LowEnergyFluidState::PrepareExplicitCarrierTransport");
    ValidateCarrierTransportConfig(config, true);
    if (m_n_pos->nGrow() < 1 || m_n_neg->nGrow() < 1 || m_ne_low->nGrow() < 1) {
        throw std::runtime_error(
            "explicit carrier transport requires low-energy fluid fields with one grow cell");
    }
    if (!m_transport_old_pos) {
        m_transport_old_pos = std::make_unique<amrex::MultiFab>(
            m_n_pos->boxArray(), m_n_pos->DistributionMap(), 1, 1);
        m_transport_old_neg = std::make_unique<amrex::MultiFab>(
            m_n_neg->boxArray(), m_n_neg->DistributionMap(), 1, 1);
        m_transport_old_ele = std::make_unique<amrex::MultiFab>(
            m_ne_low->boxArray(), m_ne_low->DistributionMap(), 1, 1);
        m_transport_den_g = std::make_unique<amrex::MultiFab>(
            m_n_pos->boxArray(), m_n_pos->DistributionMap(), 1, 1);
    }
    amrex::MultiFab::Copy(*m_transport_old_pos, *m_n_pos, 0, 0, 1, 1);
    amrex::MultiFab::Copy(*m_transport_old_neg, *m_n_neg, 0, 0, 1, 1);
    amrex::MultiFab::Copy(*m_transport_old_ele, *m_ne_low, 0, 0, 1, 1);
    if (source_dt_s > 0.0) {
        amrex::MultiFab::Saxpy(*m_transport_old_pos, source_dt_s, *m_s_positive_ion, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(*m_transport_old_ele, source_dt_s, *m_s_low_electron, 0, 0, 1, 0);
    }
    m_transport_old_pos->FillBoundary(geom.periodicity());
    m_transport_old_neg->FillBoundary(geom.periodicity());
    m_transport_old_ele->FillBoundary(geom.periodicity());
    m_transport_den_g->setVal(config.transport_density_ratio);
    if (density_ratio != nullptr) {
        amrex::MultiFab::Copy(*m_transport_den_g, *density_ratio, 0, 0, 1, 0);
    }
    m_transport_den_g->FillBoundary(geom.periodicity());
    m_explicit_transport_prepared = true;
}

LowEnergyFaceTransportDiagnostics LowEnergyFluidState::BuildExplicitFaceTransport(
    double dt,
    amrex::Geometry const& geom,
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& face_efield,
    LowEnergyFluidConfig const& config,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& face_number_flux,
    amrex::MultiFab const* electron_mobility_cell,
    amrex::MultiFab const* electron_diffusion_tensor_cell)
{
    auto const metric = fluid_metric(geom);
    double const scalar_transport_density = config.transport_density_ratio;

    BL_PROFILE("LowEnergyFluidState::BuildExplicitFaceTransport");
    if (!m_explicit_transport_prepared) {
        throw std::runtime_error(
            "BuildExplicitFaceTransport requires PrepareExplicitCarrierTransport");
    }
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::runtime_error(
            "explicit carrier face transport requires finite, positive dt");
    }
    bool const electron_mobility_from_field = config.electron_drift_enable
        && config.electron_mobility_model == "en_table_flux_v1";
    LowEnergyFaceTransportDiagnostics diagnostics;
    if (electron_mobility_from_field != (electron_mobility_cell != nullptr)
        || electron_mobility_from_field
            != (electron_diffusion_tensor_cell != nullptr)) {
        throw std::runtime_error(
            "en_table_flux_v1 face transport requires exactly its evaluated "
            "electron-mobility and diffusion-tensor cell fields");
    }
    if (electron_mobility_from_field) {
        // The face average reads the neighbor cell across box seams, so the
        // supplied field must carry filled ghost values.  Values themselves
        // are the producer's business: BuildElectronClosureCellFields counts
        // invalid cells and EvaluateClosureFields aborts collectively on them
        // before any of this is committed.
        if (electron_mobility_cell->nGrow() < 1) {
            throw std::runtime_error(
                "en_table_flux_v1 electron-mobility field needs >= 1 ghost cell");
        }
        if (electron_diffusion_tensor_cell->nGrow() < 1
            || electron_diffusion_tensor_cell->nComp() < 3) {
            throw std::runtime_error(
                "en_table_flux_v1 diffusion tensor needs 3 components and "
                ">= 1 ghost cell");
        }

        // D = D_T I + (D_L - D_T) b b^T with unit b has eigenvalues D_T and
        // D_L, both non-negative closure values, so it is positive
        // semidefinite by construction.  Its largest eigenvalue,
        // (Drr + Dzz + hypot(Drr - Dzz, 2 Drz)) / 2, supplies a
        // density-independent explicit-diffusion stability bound.
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<double> data(op);
        using Tuple = typename decltype(data)::Type;
        for (amrex::MFIter mfi(*electron_diffusion_tensor_cell);
             mfi.isValid(); ++mfi) {
            auto const tensor = electron_diffusion_tensor_cell->const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> Tuple {
                double const drr = static_cast<double>(tensor(i, j, k, 0));
                double const drz = static_cast<double>(tensor(i, j, k, 1));
                double const dzz = static_cast<double>(tensor(i, j, k, 2));
                return Tuple{0.5 * (drr + dzz + std::hypot(drr - dzz, 2.0 * drz))};
            });
        }
        double tensor_max_eigenvalue = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(tensor_max_eigenvalue);
        auto const cell_size = geom.CellSizeArray();
        double inverse_spacing_squared = 1.0 / (cell_size[0] * cell_size[0]);
        inverse_spacing_squared += 1.0 / (cell_size[1] * cell_size[1]);
        diagnostics.max_explicit_diffusion_number =
            2.0 * dt * tensor_max_eigenvalue * inverse_spacing_squared;
        if (!std::isfinite(diagnostics.max_explicit_diffusion_number)
            || diagnostics.max_explicit_diffusion_number >= 1.0) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "explicit electron diffusion violates its stability bound: "
                    << diagnostics.max_explicit_diffusion_number << " >= 1";
            throw std::runtime_error(message.str());
        }
    }
    amrex::GpuArray<MobilityLaw, 3> const mobility = {
        resolve_mobility_law(
            config.ion_mobility_model,
            config.positive_ion_mobility_m2_per_vs,
            config.positive_ion_reduced_mobility_stp_m2_per_vs,
            config.ion_mobility_density_ratio_floor),
        resolve_mobility_law(
            config.ion_mobility_model,
            config.negative_ion_mobility_m2_per_vs,
            config.negative_ion_reduced_mobility_stp_m2_per_vs,
            config.ion_mobility_density_ratio_floor),
        resolve_mobility_law(
            config.electron_mobility_model,
            config.electron_mobility_m2_per_vs,
            /*reduced=*/0.0,
            config.ion_mobility_density_ratio_floor)};
    amrex::GpuArray<bool, 3> const enabled = {
        config.ion_drift_enable,
        config.ion_drift_enable,
        config.electron_drift_enable};
    auto const domain = geom.Domain();
    auto const dx = geom.CellSizeArray();
    bool const periodic_z = geom.isPeriodic(1);

    double local_max_face_conductivity = 0.0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (face_number_flux[dir] == nullptr || face_efield[dir] == nullptr
            || face_number_flux[dir]->nComp() != 3) {
            throw std::runtime_error("invalid explicit face-transport binding");
        }
        face_number_flux[dir]->setVal(0.0);
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<double> data(op);
        using Tuple = typename decltype(data)::Type;
        for (amrex::MFIter mfi(*face_number_flux[dir], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const bx = mfi.tilebox();
            auto const pos = m_transport_old_pos->const_array(mfi);
            auto const neg = m_transport_old_neg->const_array(mfi);
            auto const ele = m_transport_old_ele->const_array(mfi);
            auto const density_ratio = m_transport_den_g->const_array(mfi);
            auto const mu_e_cell = electron_mobility_from_field
                ? electron_mobility_cell->const_array(mfi)
                : amrex::Array4<amrex::Real const>{};
            auto const diffusion_tensor = electron_mobility_from_field
                ? electron_diffusion_tensor_cell->const_array(mfi)
                : amrex::Array4<amrex::Real const>{};
            auto const e = face_efield[dir]->const_array(mfi);
            auto const flux = face_number_flux[dir]->array(mfi);
            op.eval(bx, data, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> Tuple {
                double local_max_face_conductivity = 0.0;
                auto carrier_density = [&](int species, int ii, int jj) {
                    double value = 0.0;
                    if (species == 0) {
                        value = static_cast<double>(pos(ii, jj, k));
                    } else if (species == 1) {
                        value = static_cast<double>(neg(ii, jj, k));
                    } else {
                        value = static_cast<double>(ele(ii, jj, k));
                    }
                    value = std::max(value, 0.0);
                    return value > carrier_transport_vacuum_density_m3
                        ? value
                        : 0.0;
                };
                auto carrier_mobility = [&](int species, int ii, int jj) {
                    // en_table_flux_v1: the caller-evaluated cell mobility is
                    // the single electron source for BOTH the number flux and
                    // the face conductivity below; ions keep their models.
                    if (species == 2 && electron_mobility_from_field) {
                        return std::max(
                            static_cast<double>(mu_e_cell(ii, jj, k)), 0.0);
                    }
                    return mobility[static_cast<std::size_t>(species)].At(
                        checked_density_ratio(
                            density_ratio, true, ii, jj, k,
                            scalar_transport_density));
                };

                double sigma_value = 0.0;
                for (int species = 0; species < 3; ++species) {
                    if (!enabled[species]) {
                        continue;
                    }
                    bool has_lo = true;
                    bool has_hi = true;
                    double n_lo = 0.0;
                    double n_hi = 0.0;
                    double mu_lo = 0.0;
                    double mu_hi = 0.0;
                    if (dir == 0) {
                        if (i <= domain.smallEnd(0)) {
                            has_lo = false;
                            has_hi = false;
                        } else if (i > domain.bigEnd(0)) {
                            has_hi = false;
                            n_lo = carrier_density(species, i - 1, j);
                            mu_lo = carrier_mobility(species, i - 1, j);
                        } else {
                            n_lo = carrier_density(species, i - 1, j);
                            n_hi = carrier_density(species, i, j);
                            mu_lo = carrier_mobility(species, i - 1, j);
                            mu_hi = carrier_mobility(species, i, j);
                        }
                    }
                    else if (!periodic_z && j <= domain.smallEnd(1)) {
                        has_lo = false;
                        n_hi = carrier_density(species, i, domain.smallEnd(1));
                        mu_hi = carrier_mobility(species, i, domain.smallEnd(1));
                    } else if (!periodic_z && j > domain.bigEnd(1)) {
                        has_hi = false;
                        n_lo = carrier_density(species, i, domain.bigEnd(1));
                        mu_lo = carrier_mobility(species, i, domain.bigEnd(1));
                    } else {
                        n_lo = carrier_density(species, i, j - 1);
                        n_hi = carrier_density(species, i, j);
                        mu_lo = carrier_mobility(species, i, j - 1);
                        mu_hi = carrier_mobility(species, i, j);
                    }
                    if (!has_lo && !has_hi) {
                        continue;
                    }
                    double const mu_face = has_lo && has_hi
                        ? 0.5 * (mu_lo + mu_hi)
                        : (has_lo ? mu_lo : mu_hi);
                    double const e_face = static_cast<double>(e(i, j, k));
                    double const charge_sign = species == 0 ? 1.0 : -1.0;
                    double const velocity = charge_sign * mu_face * e_face;
                    double donor_density = 0.0;
                    if (velocity > 0.0 && has_lo) {
                        donor_density = n_lo;
                    } else if (velocity < 0.0 && has_hi) {
                        donor_density = n_hi;
                    } else if (velocity == 0.0 && has_lo && has_hi) {
                        // The zero-field conductivity diagnostic has no unique
                        // upwind donor.  Physical drift flux is exactly zero.
                        donor_density = 0.5 * (n_lo + n_hi);
                    }
                    double number_flux = velocity * donor_density;
                    if (species == 2 && electron_mobility_from_field) {
                        // Field-aligned Fick flux
                        //   Gamma_diff = -[D_T I + (D_L-D_T)bb] grad(n_e).
                        // Tensor components were built at cell centers from
                        // the same E/N closure, then filled across box seams.
                        // Physical outer boundaries are absorbing (n=0 at the
                        // face); the RZ axis is reflecting by regularity.
                        double diffusive_flux = 0.0;
                        if (dir == 0) {
                            if (i > domain.smallEnd(0)
                                && i <= domain.bigEnd(0)) {
                                double const d_rr = 0.5 * (
                                    static_cast<double>(
                                        diffusion_tensor(i - 1, j, k, 0))
                                    + static_cast<double>(
                                        diffusion_tensor(i, j, k, 0)));
                                double const d_rz = 0.5 * (
                                    static_cast<double>(
                                        diffusion_tensor(i - 1, j, k, 1))
                                    + static_cast<double>(
                                        diffusion_tensor(i, j, k, 1)));
                                double const grad_r = (n_hi - n_lo) / dx[0];
                                double grad_z = 0.0;
                                auto axial_gradient = [&](int ii, int jj) {
                                    if (!periodic_z
                                        && jj <= domain.smallEnd(1)) {
                                        return (
                                            carrier_density(2, ii, jj + 1)
                                            - carrier_density(2, ii, jj))
                                            / dx[1];
                                    }
                                    if (!periodic_z
                                        && jj >= domain.bigEnd(1)) {
                                        return (
                                            carrier_density(2, ii, jj)
                                            - carrier_density(2, ii, jj - 1))
                                            / dx[1];
                                    }
                                    return (
                                        carrier_density(2, ii, jj + 1)
                                        - carrier_density(2, ii, jj - 1))
                                        / (2.0 * dx[1]);
                                };
                                grad_z = 0.5 * (
                                    axial_gradient(i - 1, j)
                                    + axial_gradient(i, j));
                                diffusive_flux = -(d_rr * grad_r + d_rz * grad_z);
                            } else if (i > domain.bigEnd(0)) {
                                double const d_rr = static_cast<double>(
                                    diffusion_tensor(i - 1, j, k, 0));
                                diffusive_flux = 2.0 * d_rr * n_lo / dx[0];
                            }
                        }
                        else if (dir == 1) {
                            if ((!periodic_z && j > domain.smallEnd(1)
                                 && j <= domain.bigEnd(1))
                                || periodic_z) {
                                double const d_rz = 0.5 * (
                                    static_cast<double>(
                                        diffusion_tensor(i, j - 1, k, 1))
                                    + static_cast<double>(
                                        diffusion_tensor(i, j, k, 1)));
                                double const d_zz = 0.5 * (
                                    static_cast<double>(
                                        diffusion_tensor(i, j - 1, k, 2))
                                    + static_cast<double>(
                                        diffusion_tensor(i, j, k, 2)));
                                auto radial_gradient = [&](int ii, int jj) {
                                    if (domain.bigEnd(0) == domain.smallEnd(0)) {
                                        return 0.0;
                                    }
                                    if (ii <= domain.smallEnd(0)) {
                                        return (
                                            carrier_density(2, ii + 1, jj)
                                            - carrier_density(2, ii, jj))
                                            / dx[0];
                                    }
                                    if (ii >= domain.bigEnd(0)) {
                                        return (
                                            carrier_density(2, ii, jj)
                                            - carrier_density(2, ii - 1, jj))
                                            / dx[0];
                                    }
                                    return (
                                        carrier_density(2, ii + 1, jj)
                                        - carrier_density(2, ii - 1, jj))
                                        / (2.0 * dx[0]);
                                };
                                double const grad_r = 0.5 * (
                                    radial_gradient(i, j - 1)
                                    + radial_gradient(i, j));
                                double const grad_z = (n_hi - n_lo) / dx[1];
                                diffusive_flux = -(d_rz * grad_r + d_zz * grad_z);
                            } else if (j <= domain.smallEnd(1)) {
                                double const d_zz = static_cast<double>(
                                    diffusion_tensor(i, domain.smallEnd(1), k, 2));
                                diffusive_flux = -2.0 * d_zz * n_hi / dx[1];
                            } else {
                                double const d_zz = static_cast<double>(
                                    diffusion_tensor(i, domain.bigEnd(1), k, 2));
                                diffusive_flux = 2.0 * d_zz * n_lo / dx[1];
                            }
                        }
                        // A cross-tensor term can point out of a transport-
                        // vacuum donor.  Such a flux has no carrier population
                        // to supply it and is therefore exactly zero.
                        if ((diffusive_flux > 0.0 && (!has_lo || n_lo <= 0.0))
                            || (diffusive_flux < 0.0
                                && (!has_hi || n_hi <= 0.0))) {
                            diffusive_flux = 0.0;
                        }
                        number_flux += diffusive_flux;
                    }
                    flux(i, j, k, species) = number_flux;
                    sigma_value += qe * mu_face * donor_density;
                }
                local_max_face_conductivity = std::max(
                    local_max_face_conductivity, sigma_value);

                return Tuple{local_max_face_conductivity};
            });
        }
        auto const values = data.value();
        local_max_face_conductivity = std::max(local_max_face_conductivity, amrex::get<0>(values));
        face_number_flux[dir]->OverrideSync(geom.periodicity());
    }
    amrex::ParallelDescriptor::ReduceRealMax(local_max_face_conductivity);
    diagnostics.max_face_conductivity_siemens_per_m = local_max_face_conductivity;
    return diagnostics;
}

void LowEnergyFluidState::ApplyExplicitCarrierTransport(
    double dt,
    amrex::Geometry const& geom,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& face_number_flux,
    LowEnergyFluidConfig const& config,
    amrex::MultiFab const* density_ratio)
{
    auto const metric = fluid_metric(geom);
    double const scalar_transport_density = config.transport_density_ratio;

    BL_PROFILE("LowEnergyFluidState::ApplyExplicitCarrierTransport");
    if (!m_explicit_transport_prepared) {
        throw std::runtime_error(
            "ApplyExplicitCarrierTransport requires PrepareExplicitCarrierTransport");
    }
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::runtime_error(
            "explicit carrier transport requires finite, positive dt");
    }
    amrex::GpuArray<bool, 3> const enabled = {
        config.ion_drift_enable,
        config.ion_drift_enable,
        config.electron_drift_enable};

    // Positivity limiter for the explicit drift-plus-diffusion density step.
    // One scale belongs to each donor cell/species. Every outgoing face uses
    // that scale, so the face stays conservative and Maxwell later consumes
    // the identical limited flux as current.
    if (!m_outflow_scale
        || m_outflow_scale->boxArray() != m_n_pos->boxArray()
        || m_outflow_scale->DistributionMap() != m_n_pos->DistributionMap()) {
        m_outflow_scale = std::make_unique<amrex::MultiFab>(
            m_n_pos->boxArray(), m_n_pos->DistributionMap(), 3, 1);
    }
    auto& outflow_scale = *m_outflow_scale;
    outflow_scale.setVal(1.0);
    double local_unlimited_max_rate = 0.0;
    double local_min_outflow_scale = 1.0;
    double local_outflow_before = 0.0;
    double local_outflow_removed = 0.0;
    double local_electron_number = 0.0;
    double local_limited_electron_number = 0.0;
    double const roundoff_factor = 512.0
        * std::numeric_limits<double>::epsilon();
    double const limiter_cfl = config.ion_drift_cfl * (1.0 - roundoff_factor);
    amrex::ReduceOps<
        amrex::ReduceOpMax, amrex::ReduceOpMin, amrex::ReduceOpSum,
        amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> limiter_op;
    amrex::ReduceData<double, double, double, double, double, double> limiter_data(limiter_op);
    using LimiterTuple = typename decltype(limiter_data)::Type;
    for (amrex::MFIter mfi(*m_n_pos, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const pos = m_transport_old_pos->const_array(mfi);
        auto const neg = m_transport_old_neg->const_array(mfi);
        auto const ele = m_transport_old_ele->const_array(mfi);
        auto const flux_r = face_number_flux[0]->const_array(mfi);
        auto const flux_z = face_number_flux[1]->const_array(mfi);
        auto const scale = outflow_scale.array(mfi);
        limiter_op.eval(bx, limiter_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> LimiterTuple {
            double local_unlimited_max_rate = 0.0;
            double local_min_outflow_scale = 1.0;
            double local_outflow_before = 0.0;
            double local_outflow_removed = 0.0;
            double local_electron_number = 0.0;
            double local_limited_electron_number = 0.0;
            auto const cell = metric.Cell(i);
            double const volume = cell.volume;
            for (int species = 0; species < 3; ++species) {
                if (!enabled[species]) {
                    continue;
                }
                double const density = std::max(
                    static_cast<double>(species == 0 ? pos(i, j, k)
                        : (species == 1 ? neg(i, j, k) : ele(i, j, k))),
                    0.0);
                double const outgoing = outgoing_number_rate(
                    flux_r, flux_z, i, j, k, species, cell);
                double const rate = density > 0.0
                    ? outgoing / (density * std::max(volume, 1.0e-300))
                    : (outgoing > 0.0
                        ? 1.0e300
                        : 0.0);
                local_unlimited_max_rate = std::max(
                    local_unlimited_max_rate, rate);
                double limiter = 1.0;
                if (outgoing > 0.0) {
                    limiter = density > 0.0
                        ? std::min(
                            1.0,
                            limiter_cfl * density * volume
                                / (dt * outgoing))
                        : 0.0;
                }
                scale(i, j, k, species) = limiter;
                local_min_outflow_scale = std::min(
                    local_min_outflow_scale, limiter);
                local_outflow_before += outgoing;
                local_outflow_removed += (1.0 - limiter) * outgoing;
                if (species == 2) {
                    double const electron_number = density * volume;
                    local_electron_number += electron_number;
                    if (limiter < 1.0) {
                        local_limited_electron_number += electron_number;
                    }
                }
            }

            return LimiterTuple{
                local_unlimited_max_rate,
                local_min_outflow_scale,
                local_outflow_before,
                local_outflow_removed,
                local_electron_number,
                local_limited_electron_number};
        });
    }
    auto const limiter_values = limiter_data.value();
    local_unlimited_max_rate = std::max(local_unlimited_max_rate, amrex::get<0>(limiter_values));
    local_min_outflow_scale = std::min(local_min_outflow_scale, amrex::get<1>(limiter_values));
    local_outflow_before += amrex::get<2>(limiter_values);
    local_outflow_removed += amrex::get<3>(limiter_values);
    local_electron_number += amrex::get<4>(limiter_values);
    local_limited_electron_number += amrex::get<5>(limiter_values);
    outflow_scale.FillBoundary(geom.periodicity());

    auto const domain = geom.Domain();
    bool const periodic_z = geom.isPeriodic(1);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto& flux_field = *face_number_flux[dir];
        for (amrex::MFIter mfi(flux_field, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const bx = mfi.tilebox();
            auto const flux = flux_field.array(mfi);
            auto const scale = outflow_scale.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                for (int species = 0; species < 3; ++species) {
                    double const value = static_cast<double>(
                        flux(i, j, k, species));
                    if (value == 0.0) {
                        continue;
                    }
                    int donor_i = i;
                    int donor_j = j;
                    bool donor_inside = true;
                    if (dir == 0) {
                        donor_i = value > 0.0 ? i - 1 : i;
                        donor_inside = donor_i >= domain.smallEnd(0)
                            && donor_i <= domain.bigEnd(0);
                    }
                    else {
                        donor_j = value > 0.0 ? j - 1 : j;
                        donor_inside = periodic_z
                            || (donor_j >= domain.smallEnd(1)
                                && donor_j <= domain.bigEnd(1));
                    }
                    flux(i, j, k, species) = donor_inside
                        ? value * static_cast<double>(
                            scale(donor_i, donor_j, k, species))
                        : 0.0;
                }
            });
        }
        flux_field.OverrideSync(geom.periodicity());
    }

    amrex::ParallelDescriptor::ReduceRealMax(local_unlimited_max_rate);
    amrex::ParallelDescriptor::ReduceRealMin(local_min_outflow_scale);
    double integrated[4] = {
        local_outflow_before,
        local_outflow_removed,
        local_electron_number,
        local_limited_electron_number};
    amrex::ParallelDescriptor::ReduceRealSum(integrated, 4);
    m_last_carrier_drift.unlimited_max_outgoing_cfl =
        dt * local_unlimited_max_rate;
    m_last_carrier_drift.min_outflow_scale = local_min_outflow_scale;
    m_last_carrier_drift.limited_outflow_fraction = integrated[0] > 0.0
        ? std::clamp(integrated[1] / integrated[0], 0.0, 1.0)
        : 0.0;
    m_last_carrier_drift.limited_electron_number_fraction = integrated[2] > 0.0
        ? std::clamp(integrated[3] / integrated[2], 0.0, 1.0)
        : 0.0;

    double local_max_outflow_rate = 0.0;
    double local_max_speed = 0.0;
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> outflow_op;
    amrex::ReduceData<double, double> outflow_data(outflow_op);
    using OutflowTuple = typename decltype(outflow_data)::Type;
    for (amrex::MFIter mfi(*m_n_pos, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const pos = m_transport_old_pos->const_array(mfi);
        auto const neg = m_transport_old_neg->const_array(mfi);
        auto const ele = m_transport_old_ele->const_array(mfi);
        auto const flux_r = face_number_flux[0]->const_array(mfi);
        auto const flux_z = face_number_flux[1]->const_array(mfi);
        outflow_op.eval(bx, outflow_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> OutflowTuple {
            double local_max_outflow_rate = 0.0;
            double local_max_speed = 0.0;
            auto const cell = metric.Cell(i);
            double const volume = cell.volume;
            for (int species = 0; species < 3; ++species) {
                if (!enabled[species]) {
                    continue;
                }
                double const density = std::max(
                    static_cast<double>(species == 0 ? pos(i, j, k)
                        : (species == 1 ? neg(i, j, k) : ele(i, j, k))),
                    0.0);
                double const outgoing = outgoing_number_rate(
                    flux_r, flux_z, i, j, k, species, cell);
                if (density > 0.0) {
                    local_max_outflow_rate = std::max(
                        local_max_outflow_rate,
                        outgoing / (density * std::max(volume, 1.0e-300)));
                    // A face flux is velocity times the upwind donor density.
                    // Only outgoing faces use this cell as that donor; dividing
                    // an incoming flux by the recipient density can report an
                    // arbitrarily large non-physical speed at a sharp front.
                    auto note_outgoing_speed = [&](double outward_flux) {
                        local_max_speed = std::max(
                            local_max_speed,
                            std::max(outward_flux, 0.0) / density);
                    };
                    note_outgoing_speed(-static_cast<double>(
                        flux_r(i, j, k, species)));
                    note_outgoing_speed(static_cast<double>(
                        flux_r(i + 1, j, k, species)));
                    note_outgoing_speed(-static_cast<double>(
                        flux_z(i, j, k, species)));
                    note_outgoing_speed(static_cast<double>(
                        flux_z(i, j + 1, k, species)));
                }
            }

            return OutflowTuple{local_max_outflow_rate, local_max_speed};
        });
    }
    auto const outflow_values = outflow_data.value();
    local_max_outflow_rate = std::max(local_max_outflow_rate, amrex::get<0>(outflow_values));
    local_max_speed = std::max(local_max_speed, amrex::get<1>(outflow_values));
    double maxima[2] = {local_max_outflow_rate, local_max_speed};
    amrex::ParallelDescriptor::ReduceRealMax(maxima, 2);
    double const max_outgoing_cfl = dt * maxima[0];
    // Every density participating in transport is either zero or at least
    // carrier_transport_vacuum_density_m3, so only ordinary floating-point
    // roundoff remains in this independently reassociated face sum.
    double const cfl_roundoff = roundoff_factor
        * std::max(1.0, config.ion_drift_cfl);
    if (max_outgoing_cfl > config.ion_drift_cfl + cfl_roundoff) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "limited full-step carrier drift violates outgoing CFL: "
                << max_outgoing_cfl << " > " << config.ion_drift_cfl;
        throw std::runtime_error(message.str());
    }
    m_last_carrier_drift.max_drift_speed_m_per_s = maxima[1];
    m_last_carrier_drift.cfl_dt_s = maxima[0] > 0.0
        ? config.ion_drift_cfl / maxima[0]
        : 1.0e300;
    m_last_carrier_drift.substeps = 1;
    m_last_carrier_drift.max_outgoing_cfl = max_outgoing_cfl;

    for (amrex::MFIter mfi(*m_n_pos, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const pos_old = m_transport_old_pos->const_array(mfi);
        auto const neg_old = m_transport_old_neg->const_array(mfi);
        auto const ele_old = m_transport_old_ele->const_array(mfi);
        auto const pos_new = m_n_pos->array(mfi);
        auto const neg_new = m_n_neg->array(mfi);
        auto const ele_new = m_ne_low->array(mfi);
        auto const rho_fluid = m_rho_fluid->array(mfi);
        auto const flux_r = face_number_flux[0]->const_array(mfi);
        auto const flux_z = face_number_flux[1]->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            auto const cell = metric.Cell(i);
            double const volume = cell.volume;
            double const shell_area = cell.axial_area;
            double const area_r_lo = cell.radial_area_lo;
            double const area_r_hi = cell.radial_area_hi;
            auto old_density = [&](int species) {
                return static_cast<double>(species == 0 ? pos_old(i, j, k)
                    : (species == 1 ? neg_old(i, j, k) : ele_old(i, j, k)));
            };
            auto updated_density = [&](int species) {
                double const r_delta =
                    static_cast<double>(flux_r(i + 1, j, k, species)) * area_r_hi
                    - static_cast<double>(flux_r(i, j, k, species)) * area_r_lo;
                double const z_delta = (
                    static_cast<double>(flux_z(i, j + 1, k, species))
                    - static_cast<double>(flux_z(i, j, k, species))) * shell_area;
                return old_density(species)
                    - dt * (r_delta + z_delta) / std::max(volume, 1.0e-300);
            };
            if (enabled[0]) {
                pos_new(i, j, k) = updated_density(0);
                neg_new(i, j, k) = updated_density(1);
            }
            if (enabled[2]) {
                ele_new(i, j, k) = updated_density(2);
            }

            auto charge_current = [&](auto const& flux, int ii, int jj) {
                return qe * (
                    static_cast<double>(flux(ii, jj, k, 0))
                    - static_cast<double>(flux(ii, jj, k, 1))
                    - static_cast<double>(flux(ii, jj, k, 2)));
            };
            double current_delta =
                charge_current(flux_r, i + 1, j) * area_r_hi
                - charge_current(flux_r, i, j) * area_r_lo;
            current_delta += (
                charge_current(flux_z, i, j + 1)
                - charge_current(flux_z, i, j)) * shell_area;
            rho_fluid(i, j, k) -=
                dt * current_delta / std::max(volume, 1.0e-300);
        });
    }

    double local_min_density = 0.0;
    double local_nonfinite_density = 0.0;
    amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax> density_op;
    amrex::ReduceData<double, double> density_data(density_op);
    using DensityTuple = typename decltype(density_data)::Type;
    for (amrex::MFIter mfi(*m_n_pos, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const pos_new = m_n_pos->const_array(mfi);
        auto const neg_new = m_n_neg->const_array(mfi);
        auto const ele_new = m_ne_low->const_array(mfi);
        density_op.eval(bx, density_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> DensityTuple {
            double local_min_density = 0.0;
            double local_nonfinite_density = 0.0;
            for (double const value : {
                     static_cast<double>(pos_new(i, j, k)),
                     static_cast<double>(neg_new(i, j, k)),
                     static_cast<double>(ele_new(i, j, k))}) {
                if (!std::isfinite(value)) {
                    local_nonfinite_density = 1.0;
                } else {
                    local_min_density = std::min(local_min_density, value);
                }
            }

            return DensityTuple{local_min_density, local_nonfinite_density};
        });
    }
    auto const density_values = density_data.value();
    local_min_density = std::min(local_min_density, amrex::get<0>(density_values));
    local_nonfinite_density = std::max(local_nonfinite_density, amrex::get<1>(density_values));
    amrex::ParallelDescriptor::ReduceRealMax(local_nonfinite_density);
    amrex::ParallelDescriptor::ReduceRealMin(local_min_density);
    if (local_nonfinite_density > 0.0) {
        throw std::runtime_error(
            "explicit carrier transport produced a non-finite density");
    }
    if (local_min_density < 0.0) {
        throw std::runtime_error(
            "explicit carrier transport produced negative density despite CFL gate");
    }
    m_explicit_transport_prepared = false;
}

void BuildElectronClosureCellFields(
    amrex::Geometry const& geom,
    amrex::MultiFab const& efield_x_cell,
    amrex::MultiFab const& efield_z_cell,
    amrex::MultiFab const* density_ratio,
    double scalar_density_ratio,
    LowEnergyElectronClosure const& closure,
    amrex::MultiFab const* electron_density,
    amrex::MultiFab const* electron_source_rate,
    amrex::MultiFab* mu_e_cell,
    amrex::MultiFab* nu_att_cell,
    amrex::MultiFab* electron_diffusion_tensor_cell,
    LowEnergyClosureFieldDiagnostics& diagnostics)
{
    auto const metric = fluid_metric(geom);
    auto const closure_view = closure.View();
    bool const write_mobility = mu_e_cell != nullptr;
    bool const write_attachment = nu_att_cell != nullptr;
    bool const write_diffusion = electron_diffusion_tensor_cell != nullptr;

    BL_PROFILE("rrea::BuildElectronClosureCellFields");
    if (!closure.Loaded()) {
        throw std::runtime_error(
            "BuildElectronClosureCellFields requires a loaded electron closure");
    }
    if (mu_e_cell == nullptr && nu_att_cell == nullptr
        && electron_diffusion_tensor_cell == nullptr) {
        throw std::runtime_error(
            "BuildElectronClosureCellFields needs at least one output field");
    }
    for (amrex::MultiFab* field : {mu_e_cell, nu_att_cell}) {
        if (field == nullptr) {
            continue;
        }
        if (field->nGrow() < 1) {
            throw std::runtime_error(
                "closure cell fields need >= 1 ghost cell for the face stencil");
        }
        // Defines the never-read out-of-domain ghosts so downstream
        // finite-value validation sees a fully initialized field.  Ghosts
        // only: the kernel writes every valid cell on every branch.
        field->setBndry(0.0);
        // Neither output is restricted to one component; the kernel writes
        // only component zero.
        if (field->nComp() > 1) {
            field->setVal(0.0, 1, field->nComp() - 1);
        }
    }
    if (write_diffusion) {
        if (electron_diffusion_tensor_cell->nGrow() < 1
            || electron_diffusion_tensor_cell->nComp() < 3) {
            throw std::runtime_error(
                "electron diffusion tensor needs 3 components and >= 1 ghost cell");
        }
        electron_diffusion_tensor_cell->setBndry(0.0);
        // The contract accepts nComp() >= 3; the kernel writes only three.
        if (electron_diffusion_tensor_cell->nComp() > 3) {
            electron_diffusion_tensor_cell->setVal(
                0.0, 3, electron_diffusion_tensor_cell->nComp() - 3);
        }
    }
    // One reduction per invocation. SetDeviceSync(true) makes the iterator
    // complete its streams when destroyed, so .value() after the loop is safe
    // inside a no-sync region.
    amrex::MFItInfo closure_iter_info;
    closure_iter_info.SetTiling(amrex::TilingIfNotGPU()).SetDeviceSync(true);
    ClosureOps op;
    ClosureData data(op);
    bool have_local_cells = false;
    for (amrex::MFIter mfi(efield_z_cell, closure_iter_info);
         mfi.isValid(); ++mfi) {
        have_local_cells = true;
        auto const bx = mfi.tilebox();
        auto const er = efield_x_cell.const_array(mfi);
        auto const ez = efield_z_cell.const_array(mfi);
        auto const den = density_ratio
            ? density_ratio->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        bool const has_density = density_ratio != nullptr;
        auto const ne = electron_density
            ? electron_density->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        auto const source = electron_source_rate
            ? electron_source_rate->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        bool const has_electron_density = electron_density != nullptr;
        bool const has_electron_source = electron_source_rate != nullptr;
        auto const mu = mu_e_cell != nullptr
            ? mu_e_cell->array(mfi)
            : amrex::Array4<amrex::Real>{};
        auto const nu = nu_att_cell != nullptr
            ? nu_att_cell->array(mfi)
            : amrex::Array4<amrex::Real>{};
        auto const diffusion = electron_diffusion_tensor_cell != nullptr
            ? electron_diffusion_tensor_cell->array(mfi)
            : amrex::Array4<amrex::Real>{};
        using Tuple = ClosureTuple;
        op.eval(bx, data, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> Tuple {
            LowEnergyClosureFieldDiagnostics diagnostics;
            // A cell the closure cannot evaluate still owes a defined value.
            auto const clear_outputs = [&]() {
                if (write_mobility) { mu(i, j, k) = 0.0; }
                if (write_attachment) { nu(i, j, k) = 0.0; }
                if (write_diffusion) {
                    diffusion(i, j, k, 0) = diffusion(i, j, k, 1)
                        = diffusion(i, j, k, 2) = 0.0;
                }
            };
            double const chi = has_density
                ? static_cast<double>(den(i, j, k))
                : scalar_density_ratio;
            double const e_mag = std::hypot(
                static_cast<double>(er(i, j, k)),
                static_cast<double>(ez(i, j, k)));
            double const electron_value = has_electron_density
                ? static_cast<double>(ne(i, j, k))
                : 0.0;
            double const source_value = has_electron_source
                ? static_cast<double>(source(i, j, k))
                : 0.0;
            if (!std::isfinite(chi) || !(chi > 0.0)
                || !std::isfinite(e_mag)
                || !std::isfinite(electron_value)
                || !std::isfinite(source_value)) {
                ++diagnostics.invalid_cells;
                clear_outputs();
                return closure_tuple(diagnostics);
            }
            // The complete local |E| and the actual positive neutral density
            // -- never a face-normal component, never the mobility floor.
            double const en_td = reduced_field_td(e_mag, chi);
            RreaClosureRangeStatus status{};
            auto const coefficients = closure_view.Evaluate(en_td, status);
            bool const held_below =
                status == RreaClosureRangeStatus::HeldBelow;
            switch (status) {
            case RreaClosureRangeStatus::HeldBelow:
                ++diagnostics.held_below_cells;
                break;
            case RreaClosureRangeStatus::AboveRange:
                ++diagnostics.above_range_cells;
                break;
            case RreaClosureRangeStatus::Invalid:
                ++diagnostics.invalid_cells;
                clear_outputs();
                return closure_tuple(diagnostics);
            case RreaClosureRangeStatus::InRange:
                break;
            }
            diagnostics.min_en_td = std::min(diagnostics.min_en_td, en_td);
            diagnostics.max_en_td = std::max(diagnostics.max_en_td, en_td);
            double const local_mu =
                static_cast<double>(coefficients.k0_flux_ref_m2_per_vs) / chi;
            double const volume = metric.CellVolume(i);
            double const electron_count =
                std::max(electron_value, 0.0) * volume;
            double const source_rate =
                std::max(source_value, 0.0) * volume;
            double const conductivity_proxy = local_mu * electron_count;
            double const electron_density_positive =
                std::max(electron_value, 0.0);
            double const source_rate_density =
                std::max(source_value, 0.0);
            double const conductivity_proxy_density =
                local_mu * electron_density_positive;
            if (status == RreaClosureRangeStatus::AboveRange
                && en_td > diagnostics.max_above_range_en_td) {
                diagnostics.max_above_range_en_td = en_td;
                diagnostics.max_above_range_i = i;
                diagnostics.max_above_range_j = j;
                diagnostics.max_above_range_electron_density_m3 =
                    electron_density_positive;
                diagnostics.max_above_range_source_rate_m3_s =
                    source_rate_density;
                diagnostics.max_above_range_conductivity_proxy_per_m_v_s =
                    conductivity_proxy_density;
            }
            diagnostics.electron_count += electron_count;
            diagnostics.electron_source_rate_per_s += source_rate;
            diagnostics.electron_conductivity_proxy_m2_per_v_s +=
                conductivity_proxy;
            diagnostics.max_electron_density_m3 = std::max(
                diagnostics.max_electron_density_m3,
                electron_density_positive);
            diagnostics.max_electron_source_rate_m3_s = std::max(
                diagnostics.max_electron_source_rate_m3_s,
                source_rate_density);
            diagnostics.max_electron_conductivity_proxy_per_m_v_s = std::max(
                diagnostics.max_electron_conductivity_proxy_per_m_v_s,
                conductivity_proxy_density);
            if (held_below) {
                diagnostics.held_below_electron_count += electron_count;
                diagnostics.held_below_electron_source_rate_per_s +=
                    source_rate;
                diagnostics.held_below_electron_conductivity_proxy_m2_per_v_s +=
                    conductivity_proxy;
                diagnostics.max_held_below_electron_density_m3 = std::max(
                    diagnostics.max_held_below_electron_density_m3,
                    electron_density_positive);
                diagnostics.max_held_below_electron_source_rate_m3_s =
                    std::max(
                        diagnostics.max_held_below_electron_source_rate_m3_s,
                        source_rate_density);
                diagnostics.max_held_below_electron_conductivity_proxy_per_m_v_s =
                    std::max(
                        diagnostics
                            .max_held_below_electron_conductivity_proxy_per_m_v_s,
                        conductivity_proxy_density);
            }
            if (write_mobility) {
                mu(i, j, k) = static_cast<amrex::Real>(local_mu);
            }
            if (write_attachment) {
                nu(i, j, k) = static_cast<amrex::Real>(
                    static_cast<double>(coefficients.nu2_ref_per_s) * chi
                    + static_cast<double>(coefficients.nu3_ref_per_s)
                        * chi * chi);
            }
            if (write_diffusion) {
                double const d_l = static_cast<double>(
                    coefficients.longitudinal_diffusion_ref_m2_per_s) / chi;
                double const d_t = static_cast<double>(
                    coefficients.transverse_diffusion_ref_m2_per_s) / chi;
                double b_r = 0.0;
                double b_z = 0.0;
                if (e_mag > 0.0) {
                    b_r = static_cast<double>(er(i, j, k)) / e_mag;
                    b_z = static_cast<double>(ez(i, j, k)) / e_mag;
                }
                double const anisotropy = d_l - d_t;
                diffusion(i, j, k, 0) = static_cast<amrex::Real>(
                    d_t + anisotropy * b_r * b_r);
                diffusion(i, j, k, 1) = static_cast<amrex::Real>(
                    anisotropy * b_r * b_z);
                diffusion(i, j, k, 2) = static_cast<amrex::Real>(
                    d_t + anisotropy * b_z * b_z);
            }

            return closure_tuple(diagnostics);
        });
    }
    // Finalize once, after the iterator has been destroyed and its streams
    // completed; the host folds merge into the caller's diagnostics.
    auto const values = data.value();
    if (have_local_cells) {
        diagnostics.min_en_td = std::min(diagnostics.min_en_td, amrex::get<0>(values));
        diagnostics.max_en_td = std::max(diagnostics.max_en_td, amrex::get<1>(values));
        diagnostics.held_below_cells += amrex::get<2>(values);
        diagnostics.above_range_cells += amrex::get<3>(values);
        diagnostics.invalid_cells += amrex::get<4>(values);
        diagnostics.electron_count += amrex::get<6>(values);
        diagnostics.held_below_electron_count += amrex::get<7>(values);
        diagnostics.electron_source_rate_per_s += amrex::get<8>(values);
        diagnostics.held_below_electron_source_rate_per_s += amrex::get<9>(values);
        diagnostics.electron_conductivity_proxy_m2_per_v_s += amrex::get<10>(values);
        diagnostics.held_below_electron_conductivity_proxy_m2_per_v_s += amrex::get<11>(values);
        diagnostics.max_electron_density_m3 = std::max(
            diagnostics.max_electron_density_m3, amrex::get<12>(values));
        diagnostics.max_held_below_electron_density_m3 = std::max(
            diagnostics.max_held_below_electron_density_m3, amrex::get<13>(values));
        diagnostics.max_electron_source_rate_m3_s = std::max(
            diagnostics.max_electron_source_rate_m3_s, amrex::get<14>(values));
        diagnostics.max_held_below_electron_source_rate_m3_s = std::max(
            diagnostics.max_held_below_electron_source_rate_m3_s, amrex::get<15>(values));
        diagnostics.max_electron_conductivity_proxy_per_m_v_s = std::max(
            diagnostics.max_electron_conductivity_proxy_per_m_v_s, amrex::get<16>(values));
        diagnostics.max_held_below_electron_conductivity_proxy_per_m_v_s = std::max(
            diagnostics.max_held_below_electron_conductivity_proxy_per_m_v_s, amrex::get<17>(values));
        double const above_max = amrex::get<5>(values);
        if (above_max > diagnostics.max_above_range_en_td) {
            // Second pass, entered only when this invocation raises the
            // above-range maximum, so ordinary in-range runs allocate nothing
            // here. Ties go to the first qualifying tile, then the first cell
            // in (k,j,i) order.
            bool located = false;
            int wi = 0;
            int wj = 0;
            int wk = 0;
            double witness_chi = scalar_density_ratio;
            double witness_ev = 0.0;
            double witness_sv = 0.0;
            for (amrex::MFIter mfi(efield_z_cell, closure_iter_info);
                 mfi.isValid() && !located; ++mfi) {
                auto const bx = mfi.tilebox();
                auto const er = efield_x_cell.const_array(mfi);
                auto const ez = efield_z_cell.const_array(mfi);
                auto const den = density_ratio
                    ? density_ratio->const_array(mfi)
                    : amrex::Array4<amrex::Real const>{};
                bool const has_density = density_ratio != nullptr;
                auto const ne = electron_density
                    ? electron_density->const_array(mfi)
                    : amrex::Array4<amrex::Real const>{};
                auto const source = electron_source_rate
                    ? electron_source_rate->const_array(mfi)
                    : amrex::Array4<amrex::Real const>{};
                bool const has_electron_density = electron_density != nullptr;
                bool const has_electron_source = electron_source_rate != nullptr;
                auto const lo = bx.smallEnd();
                auto const len = bx.length();
                amrex::ReduceOps<amrex::ReduceOpMin> location_op;
                amrex::ReduceData<amrex::Long> location_data(location_op);
                using Location = typename decltype(location_data)::Type;
                location_op.eval(bx, location_data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> Location {
                    double const chi = has_density ? static_cast<double>(den(i,j,k))
                                                   : scalar_density_ratio;
                    double const mag = std::hypot(static_cast<double>(er(i,j,k)),
                                                 static_cast<double>(ez(i,j,k)));
                    double const ev = has_electron_density ? static_cast<double>(ne(i,j,k)) : 0.0;
                    double const sv = has_electron_source ? static_cast<double>(source(i,j,k)) : 0.0;
                    double const en = reduced_field_td(mag, chi);
                    if (chi > 0.0 && std::isfinite(chi) && std::isfinite(mag)
                        && std::isfinite(ev) && std::isfinite(sv) && en == above_max) {
                        amrex::Long q = i - lo[0];
                        q += static_cast<amrex::Long>(len[0]) * (j - lo[1]);
                        return {q};
                    }
                    return {std::numeric_limits<amrex::Long>::max()};
                });
                auto q = amrex::get<0>(location_data.value());
                if (q == std::numeric_limits<amrex::Long>::max()) {
                    continue;
                }
                wi = lo[0] + static_cast<int>(q % len[0]);
                q /= len[0];
                wj = lo[1] + static_cast<int>(q % len[1]);
                q /= len[1];
                witness_chi = has_density ? host_cell(den, wi, wj, wk) : scalar_density_ratio;
                witness_ev = has_electron_density
                    ? std::max(host_cell(ne, wi, wj, wk), 0.0) : 0.0;
                witness_sv = has_electron_source
                    ? std::max(host_cell(source, wi, wj, wk), 0.0) : 0.0;
                located = true;
            }
            if (!located) {
                throw std::runtime_error("cannot locate the above-range closure cell");
            }
            RreaClosureRangeStatus status;
            auto const coeff = closure.Evaluate(above_max, status);
            diagnostics.max_above_range_en_td = above_max;
            diagnostics.max_above_range_i = wi;
            diagnostics.max_above_range_j = wj;
            diagnostics.max_above_range_electron_density_m3 = witness_ev;
            diagnostics.max_above_range_source_rate_m3_s = witness_sv;
            diagnostics.max_above_range_conductivity_proxy_per_m_v_s =
                static_cast<double>(coeff.k0_flux_ref_m2_per_vs) / witness_chi * witness_ev;
        }
    }
    for (amrex::MultiFab* field : {mu_e_cell, nu_att_cell}) {
        if (field != nullptr) {
            field->FillBoundary(geom.periodicity());
        }
    }
    if (write_diffusion) {
        electron_diffusion_tensor_cell->FillBoundary(geom.periodicity());
    }
}

} // namespace rrea
