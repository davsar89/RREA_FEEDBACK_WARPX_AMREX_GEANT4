#include "rrea/LowEnergyFluidState.H"

#include "rrea/RreaConstants.H"
#include "rrea/RreaFieldHandoff.H"
#include "rrea/RreaInteractionTables.H"

#include <AMReX_Array4.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

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

double fluid_cell_volume_rz(amrex::Geometry const& geom, int i)
{
    auto const dx = geom.CellSizeArray();
    double const r_lo = geom.ProbLo(0)
        + static_cast<double>(i - geom.Domain().smallEnd(0)) * dx[0];
#if (AMREX_SPACEDIM >= 2)
    return rz_shell_volume(r_lo, r_lo + dx[0], dx[1]);
#else
    return rz_shell_volume(r_lo, r_lo + dx[0], 1.0);
#endif
}

double fluid_r_center(amrex::Geometry const& geom, int i)
{
    int const radial_lo = geom.Domain().smallEnd(0);
    return geom.ProbLo(0)
        + (static_cast<double>(i - radial_lo) + 0.5) * geom.CellSize(0);
}

double fluid_z_center(amrex::Geometry const& geom, int j)
{
#if (AMREX_SPACEDIM >= 2)
    int const axial_lo = geom.Domain().smallEnd(1);
    return geom.ProbLo(1)
        + (static_cast<double>(j - axial_lo) + 0.5) * geom.CellSize(1);
#else
    amrex::ignore_unused(geom, j);
    return 0.0;
#endif
}

void audit_pre_sync_face_copies(
    amrex::MultiFab const& field,
    amrex::Geometry const& geom,
    LowEnergyFaceSyncDiagnostics& diagnostics)
{
    amrex::MultiFab canonical(
        field.boxArray(), field.DistributionMap(), field.nComp(), 0);
    amrex::MultiFab::Copy(canonical, field, 0, 0, field.nComp(), 0);
    canonical.OverrideSync(geom.periodicity());
    double local_max_difference = 0.0;
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const original = field.const_array(mfi);
        auto const synchronized = canonical.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            for (int component = 0; component < field.nComp(); ++component) {
                local_max_difference = std::max(
                    local_max_difference,
                    std::abs(
                        static_cast<double>(original(i, j, k, component))
                        - static_cast<double>(synchronized(i, j, k, component))));
            }
        });
    }
    amrex::ParallelDescriptor::ReduceRealMax(local_max_difference);
    diagnostics.max_pre_sync_duplicate_difference = std::max(
        diagnostics.max_pre_sync_duplicate_difference, local_max_difference);
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
    amrex::MultiFab const* density_ratio)
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

        // D must be symmetric positive-semidefinite.  For the stored
        // (Drr, Drz, Dzz) representation this is Drr,Dzz >= 0 and
        // Drr*Dzz-Drz^2 >= 0.  Its largest eigenvalue supplies a
        // density-independent explicit-diffusion stability bound.
        double tensor_extrema[2] = {0.0, 0.0}; // invalid flag, max eigenvalue
        for (amrex::MFIter mfi(*electron_diffusion_tensor_cell);
             mfi.isValid(); ++mfi) {
            auto const bx = mfi.validbox();
            auto const tensor = electron_diffusion_tensor_cell->const_array(mfi);
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
                double const drr = static_cast<double>(tensor(i, j, k, 0));
                double const drz = static_cast<double>(tensor(i, j, k, 1));
                double const dzz = static_cast<double>(tensor(i, j, k, 2));
                double const determinant = drr * dzz - drz * drz;
                double const determinant_scale = std::max(
                    {std::abs(drr * dzz), std::abs(drz * drz), 1.0});
                double const determinant_tolerance = 512.0
                    * std::numeric_limits<double>::epsilon()
                    * determinant_scale;
                if (!std::isfinite(drr) || !std::isfinite(drz)
                    || !std::isfinite(dzz) || drr < 0.0 || dzz < 0.0
                    || determinant < -determinant_tolerance) {
                    tensor_extrema[0] = 1.0;
                    return;
                }
                double const discriminant = std::hypot(drr - dzz, 2.0 * drz);
                tensor_extrema[1] = std::max(
                    tensor_extrema[1], 0.5 * (drr + dzz + discriminant));
            });
        }
        amrex::ParallelDescriptor::ReduceRealMax(tensor_extrema, 2);
        if (tensor_extrema[0] > 0.0) {
            throw std::runtime_error(
                "en_table_flux_v1 diffusion tensor is not finite symmetric "
                "positive-semidefinite");
        }
        auto const cell_size = geom.CellSizeArray();
        double inverse_spacing_squared = 1.0 / (cell_size[0] * cell_size[0]);
#if (AMREX_SPACEDIM >= 2)
        inverse_spacing_squared += 1.0 / (cell_size[1] * cell_size[1]);
#endif
        diagnostics.max_explicit_diffusion_number =
            2.0 * dt * tensor_extrema[1] * inverse_spacing_squared;
        if (!std::isfinite(diagnostics.max_explicit_diffusion_number)
            || diagnostics.max_explicit_diffusion_number >= 1.0) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "explicit electron diffusion violates its stability bound: "
                    << diagnostics.max_explicit_diffusion_number << " >= 1";
            throw std::runtime_error(message.str());
        }
    }
    std::array<MobilityLaw, 3> const mobility = {
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
    bool const enabled[3] = {
        config.ion_drift_enable,
        config.ion_drift_enable,
        config.electron_drift_enable};
    auto const domain = geom.Domain();
    auto const dx = geom.CellSizeArray();
#if (AMREX_SPACEDIM >= 2)
    bool const periodic_z = geom.isPeriodic(1);
#endif

    double local_max_face_conductivity = 0.0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (face_number_flux[dir] == nullptr || face_efield[dir] == nullptr
            || face_number_flux[dir]->nComp() != 3) {
            throw std::runtime_error("invalid explicit face-transport binding");
        }
        face_number_flux[dir]->setVal(0.0);
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
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
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
                            config.transport_density_ratio));
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
#if (AMREX_SPACEDIM >= 2)
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
#endif
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
#if (AMREX_SPACEDIM >= 2)
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
#endif
                                diffusive_flux = -(d_rr * grad_r + d_rz * grad_z);
                            } else if (i > domain.bigEnd(0)) {
                                double const d_rr = static_cast<double>(
                                    diffusion_tensor(i - 1, j, k, 0));
                                diffusive_flux = 2.0 * d_rr * n_lo / dx[0];
                            }
                        }
#if (AMREX_SPACEDIM >= 2)
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
#endif
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
            });
        }
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
    amrex::MultiFab const* density_ratio,
    bool collect_center_diagnostics,
    LowEnergyFaceSyncDiagnostics* face_sync_diagnostics)
{
    BL_PROFILE("LowEnergyFluidState::ApplyExplicitCarrierTransport");
    if (!m_explicit_transport_prepared) {
        throw std::runtime_error(
            "ApplyExplicitCarrierTransport requires PrepareExplicitCarrierTransport");
    }
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::runtime_error(
            "explicit carrier transport requires finite, positive dt");
    }
    bool const enabled[3] = {
        config.ion_drift_enable,
        config.ion_drift_enable,
        config.electron_drift_enable};
    auto const dx = geom.CellSizeArray();

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
    for (amrex::MFIter mfi(*m_n_pos, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const pos = m_transport_old_pos->const_array(mfi);
        auto const neg = m_transport_old_neg->const_array(mfi);
        auto const ele = m_transport_old_ele->const_array(mfi);
        auto const flux_r = face_number_flux[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
        auto const flux_z = face_number_flux[1]->const_array(mfi);
#endif
        auto const scale = outflow_scale.array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            int const radial_lo = geom.Domain().smallEnd(0);
            double const r_lo = geom.ProbLo(0)
                + static_cast<double>(i - radial_lo) * dx[0];
            double const r_hi = r_lo + dx[0];
#if (AMREX_SPACEDIM >= 2)
            double const dz = dx[1];
#else
            double const dz = 1.0;
#endif
            double const volume = fluid_cell_volume_rz(geom, i);
            double const shell_area = pi * std::max(
                r_hi * r_hi - r_lo * r_lo, 0.0);
            double const area_r_lo = 2.0 * pi * r_lo * dz;
            double const area_r_hi = 2.0 * pi * r_hi * dz;
            for (int species = 0; species < 3; ++species) {
                if (!enabled[species]) {
                    continue;
                }
                double const density = std::max(
                    static_cast<double>(species == 0 ? pos(i, j, k)
                        : (species == 1 ? neg(i, j, k) : ele(i, j, k))),
                    0.0);
                double outgoing =
                    std::max(static_cast<double>(
                        flux_r(i + 1, j, k, species)), 0.0) * area_r_hi
                    + std::max(-static_cast<double>(
                        flux_r(i, j, k, species)), 0.0) * area_r_lo;
#if (AMREX_SPACEDIM >= 2)
                outgoing += (
                    std::max(static_cast<double>(
                        flux_z(i, j + 1, k, species)), 0.0)
                    + std::max(-static_cast<double>(
                        flux_z(i, j, k, species)), 0.0)) * shell_area;
#endif
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
        });
    }
    outflow_scale.FillBoundary(geom.periodicity());

    auto const domain = geom.Domain();
#if (AMREX_SPACEDIM >= 2)
    bool const periodic_z = geom.isPeriodic(1);
#endif
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto& flux_field = *face_number_flux[dir];
        for (amrex::MFIter mfi(flux_field, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const bx = mfi.tilebox();
            auto const flux = flux_field.array(mfi);
            auto const scale = outflow_scale.const_array(mfi);
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
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
#if (AMREX_SPACEDIM >= 2)
                    else {
                        donor_j = value > 0.0 ? j - 1 : j;
                        donor_inside = periodic_z
                            || (donor_j >= domain.smallEnd(1)
                                && donor_j <= domain.bigEnd(1));
                    }
#endif
                    flux(i, j, k, species) = donor_inside
                        ? value * static_cast<double>(
                            scale(donor_i, donor_j, k, species))
                        : 0.0;
                }
            });
        }
        if (face_sync_diagnostics != nullptr) {
            audit_pre_sync_face_copies(
                flux_field, geom, *face_sync_diagnostics);
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
    for (amrex::MFIter mfi(*m_n_pos, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const pos = m_transport_old_pos->const_array(mfi);
        auto const neg = m_transport_old_neg->const_array(mfi);
        auto const ele = m_transport_old_ele->const_array(mfi);
        auto const flux_r = face_number_flux[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
        auto const flux_z = face_number_flux[1]->const_array(mfi);
#endif
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            int const radial_lo = geom.Domain().smallEnd(0);
            double const r_lo = geom.ProbLo(0)
                + static_cast<double>(i - radial_lo) * dx[0];
            double const r_hi = r_lo + dx[0];
#if (AMREX_SPACEDIM >= 2)
            double const dz = dx[1];
#else
            double const dz = 1.0;
#endif
            double const volume = fluid_cell_volume_rz(geom, i);
            double const shell_area = pi * std::max(r_hi * r_hi - r_lo * r_lo, 0.0);
            double const area_r_lo = 2.0 * pi * r_lo * dz;
            double const area_r_hi = 2.0 * pi * r_hi * dz;
            for (int species = 0; species < 3; ++species) {
                if (!enabled[species]) {
                    continue;
                }
                double const density = std::max(
                    static_cast<double>(species == 0 ? pos(i, j, k)
                        : (species == 1 ? neg(i, j, k) : ele(i, j, k))),
                    0.0);
                double outgoing =
                    std::max(static_cast<double>(flux_r(i + 1, j, k, species)), 0.0)
                        * area_r_hi
                    + std::max(-static_cast<double>(flux_r(i, j, k, species)), 0.0)
                        * area_r_lo;
#if (AMREX_SPACEDIM >= 2)
                outgoing += (
                    std::max(static_cast<double>(flux_z(i, j + 1, k, species)), 0.0)
                    + std::max(-static_cast<double>(flux_z(i, j, k, species)), 0.0))
                    * shell_area;
#endif
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
#if (AMREX_SPACEDIM >= 2)
                    note_outgoing_speed(-static_cast<double>(
                        flux_z(i, j, k, species)));
                    note_outgoing_speed(static_cast<double>(
                        flux_z(i, j + 1, k, species)));
#endif
                }
            }
        });
    }
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
#if (AMREX_SPACEDIM >= 2)
        auto const flux_z = face_number_flux[1]->const_array(mfi);
#endif
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            int const radial_lo = geom.Domain().smallEnd(0);
            double const r_lo = geom.ProbLo(0)
                + static_cast<double>(i - radial_lo) * dx[0];
            double const r_hi = r_lo + dx[0];
#if (AMREX_SPACEDIM >= 2)
            double const dz = dx[1];
#else
            double const dz = 1.0;
#endif
            double const volume = fluid_cell_volume_rz(geom, i);
            double const shell_area = pi * std::max(r_hi * r_hi - r_lo * r_lo, 0.0);
            double const area_r_lo = 2.0 * pi * r_lo * dz;
            double const area_r_hi = 2.0 * pi * r_hi * dz;
            auto old_density = [&](int species) {
                return static_cast<double>(species == 0 ? pos_old(i, j, k)
                    : (species == 1 ? neg_old(i, j, k) : ele_old(i, j, k)));
            };
            auto updated_density = [&](int species) {
                double const r_delta =
                    static_cast<double>(flux_r(i + 1, j, k, species)) * area_r_hi
                    - static_cast<double>(flux_r(i, j, k, species)) * area_r_lo;
#if (AMREX_SPACEDIM >= 2)
                double const z_delta = (
                    static_cast<double>(flux_z(i, j + 1, k, species))
                    - static_cast<double>(flux_z(i, j, k, species))) * shell_area;
#else
                double const z_delta = 0.0;
#endif
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
#if (AMREX_SPACEDIM >= 2)
            current_delta += (
                charge_current(flux_z, i, j + 1)
                - charge_current(flux_z, i, j)) * shell_area;
#endif
            rho_fluid(i, j, k) -=
                dt * current_delta / std::max(volume, 1.0e-300);
        });
    }

    double local_min_density = 0.0;
    double local_nonfinite_density = 0.0;
    for (amrex::MFIter mfi(*m_n_pos, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const pos_new = m_n_pos->const_array(mfi);
        auto const neg_new = m_n_neg->const_array(mfi);
        auto const ele_new = m_ne_low->const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
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
        });
    }
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
    CollectCarrierDriftCenterDiagnostics(
        geom, config, density_ratio, collect_center_diagnostics);
}

void LowEnergyFluidState::CollectCarrierDriftCenterDiagnostics(
    amrex::Geometry const& geom,
    LowEnergyFluidConfig const& config,
    amrex::MultiFab const* density_ratio,
    bool collect_center_diagnostics)
{
    if (!collect_center_diagnostics && m_positive_ion_initial_center_set) {
        return;
    }
    BL_PROFILE("LowEnergyFluidState::CollectCarrierDriftCenterDiagnostics");
    MobilityLaw const positive_mobility = resolve_mobility_law(
        config.ion_mobility_model,
        config.positive_ion_mobility_m2_per_vs,
        config.positive_ion_reduced_mobility_stp_m2_per_vs,
        config.ion_mobility_density_ratio_floor);
    double pos_count = 0.0;
    double neg_count = 0.0;
    double pos_r = 0.0;
    double pos_z = 0.0;
    double neg_r = 0.0;
    double neg_z = 0.0;
    double pos_density_ratio = 0.0;
    double pos_mu = 0.0;
    for (amrex::MFIter mfi(*m_n_pos, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const pos = m_n_pos->const_array(mfi);
        auto const neg = m_n_neg->const_array(mfi);
        auto const den = density_ratio
            ? density_ratio->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        bool const has_density = density_ratio != nullptr;
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            double const volume = fluid_cell_volume_rz(geom, i);
            double const p = std::max(static_cast<double>(pos(i, j, k)), 0.0) * volume;
            double const n = std::max(static_cast<double>(neg(i, j, k)), 0.0) * volume;
            pos_count += p;
            neg_count += n;
            pos_r += p * fluid_r_center(geom, i);
            pos_z += p * fluid_z_center(geom, j);
            neg_r += n * fluid_r_center(geom, i);
            neg_z += n * fluid_z_center(geom, j);
            double const dens = has_density
                ? std::max(static_cast<double>(den(i, j, k)), 1.0e-300)
                : config.transport_density_ratio;
            pos_density_ratio += p * dens;
            pos_mu += p * positive_mobility.At(dens);
        });
    }
    double sums[8] = {
        pos_count, neg_count, pos_r, pos_z,
        neg_r, neg_z, pos_density_ratio, pos_mu};
    amrex::ParallelDescriptor::ReduceRealSum(sums, 8);
    pos_count = sums[0];
    neg_count = sums[1];
    if (pos_count > 0.0) {
        m_last_carrier_drift.positive_ion_center_r_m = sums[2] / pos_count;
        m_last_carrier_drift.positive_ion_center_z_m = sums[3] / pos_count;
        if (!m_positive_ion_initial_center_set) {
            m_positive_ion_initial_center_z_m =
                m_last_carrier_drift.positive_ion_center_z_m;
            m_positive_ion_initial_center_set = true;
        }
        m_last_carrier_drift.positive_ion_initial_center_z_m =
            m_positive_ion_initial_center_z_m;
        m_last_carrier_drift.positive_ion_drift_z_m =
            m_last_carrier_drift.positive_ion_center_z_m
            - m_positive_ion_initial_center_z_m;
        m_last_carrier_drift.density_ratio_at_positive_ion_center =
            sums[6] / pos_count;
        m_last_carrier_drift.mu_pos_at_positive_ion_center_m2_per_vs =
            sums[7] / pos_count;
    }
    if (neg_count > 0.0) {
        m_last_carrier_drift.negative_ion_center_r_m = sums[4] / neg_count;
        m_last_carrier_drift.negative_ion_center_z_m = sums[5] / neg_count;
    }
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
        // finite-value validation sees a fully initialized field.
        field->setVal(0.0);
    }
    if (electron_diffusion_tensor_cell != nullptr) {
        if (electron_diffusion_tensor_cell->nGrow() < 1
            || electron_diffusion_tensor_cell->nComp() < 3) {
            throw std::runtime_error(
                "electron diffusion tensor needs 3 components and >= 1 ghost cell");
        }
        electron_diffusion_tensor_cell->setVal(0.0);
    }
    for (amrex::MFIter mfi(efield_z_cell, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
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
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
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
                return;
            }
            // The complete local |E| and the actual positive neutral density
            // -- never a face-normal component, never the mobility floor.
            double const en_td =
                e_mag * 1.0e21 / (chi * n_loschmidt_m3);
            RreaClosureRangeStatus status{};
            auto const coefficients = closure.Evaluate(en_td, status);
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
                return;
            case RreaClosureRangeStatus::InRange:
                break;
            }
            diagnostics.min_en_td = std::min(diagnostics.min_en_td, en_td);
            diagnostics.max_en_td = std::max(diagnostics.max_en_td, en_td);
            double const local_mu =
                static_cast<double>(coefficients.k0_flux_ref_m2_per_vs) / chi;
            double const volume = fluid_cell_volume_rz(geom, i);
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
            if (mu_e_cell != nullptr) {
                mu(i, j, k) = static_cast<amrex::Real>(local_mu);
            }
            if (nu_att_cell != nullptr) {
                nu(i, j, k) = static_cast<amrex::Real>(
                    static_cast<double>(coefficients.nu2_ref_per_s) * chi
                    + static_cast<double>(coefficients.nu3_ref_per_s)
                        * chi * chi);
            }
            if (electron_diffusion_tensor_cell != nullptr) {
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
        });
    }
    for (amrex::MultiFab* field : {mu_e_cell, nu_att_cell}) {
        if (field != nullptr) {
            field->FillBoundary(geom.periodicity());
        }
    }
    if (electron_diffusion_tensor_cell != nullptr) {
        electron_diffusion_tensor_cell->FillBoundary(geom.periodicity());
    }
}

} // namespace rrea
