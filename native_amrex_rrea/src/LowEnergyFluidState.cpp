#include "rrea/LowEnergyFluidState.H"

#include "rrea/RreaConstants.H"

#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rrea {

namespace {

std::unique_ptr<amrex::MultiFab> make_scalar_field(
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    int n_grow)
{
    return std::make_unique<amrex::MultiFab>(ba, dm, 1, n_grow);
}

// Exact nonnegative update for one binary pair-loss channel
//   da/dt = db/dt = -alpha*a*b.
// The difference a-b is invariant.  Returning the shared decrement lets the
// electron-ion and ion-ion channels use one reviewed equation without
// conflating their coefficients or species.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double exact_pair_loss(
    double a,
    double b,
    double alpha,
    double dt) noexcept
{
    double const small = amrex::min(a, b);
    double const large = amrex::max(a, b);
    if (!(small > 0.0) || !(alpha > 0.0) || !(dt > 0.0)) {
        return 0.0;
    }
    double const delta = large - small;
    double remaining_small;
    if (delta <= 1.0e-12 * amrex::max(large, 1.0)) {
        remaining_small = small / (1.0 + alpha * small * dt);
    } else {
        double const ratio0 = small / large;
        double const ratio = ratio0 * std::exp(-alpha * delta * dt);
        remaining_small = ratio >= 1.0
            ? small
            : delta * ratio / (1.0 - ratio);
    }
    return small - amrex::min(amrex::max(remaining_small, 0.0), small);
}

}  // namespace

LowEnergyFluidState::LowEnergyFluidState(
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    int n_grow)
    : m_ne_low(make_scalar_field(ba, dm, n_grow)),
      m_n_pos(make_scalar_field(ba, dm, n_grow)),
      m_n_neg(make_scalar_field(ba, dm, n_grow)),
      m_s_low_electron(make_scalar_field(ba, dm, n_grow)),
      m_s_positive_ion(make_scalar_field(ba, dm, n_grow)),
      m_s_direct_negative_ion(make_scalar_field(ba, dm, n_grow)),
      m_sigma(make_scalar_field(ba, dm, n_grow)),
      m_tau_m(make_scalar_field(ba, dm, n_grow)),
      m_rho_fluid(make_scalar_field(ba, dm, n_grow)),
      m_charge_residual(make_scalar_field(ba, dm, 0))
{
    SetVal(0.0);
}

void LowEnergyFluidState::SetVal(double value)
{
    m_ne_low->setVal(value);
    m_n_pos->setVal(value);
    m_n_neg->setVal(value);
    m_s_low_electron->setVal(value);
    m_s_positive_ion->setVal(value);
    m_s_direct_negative_ion->setVal(value);
    m_sigma->setVal(value);
    m_tau_m->setVal(value);
    m_rho_fluid->setVal(value);
    m_charge_residual->setVal(0.0);
    m_last_carrier_drift = LowEnergyCarrierDriftDiagnostics{};
    m_last_reaction_charge_residual = 0.0;
    m_positive_ion_initial_center_set = false;
    m_positive_ion_initial_center_z_m = 0.0;
    m_explicit_transport_prepared = false;
}

void LowEnergyFluidState::UpdateFromIonizationSource(
    double dt,
    LowEnergyFluidConfig const& config,
    amrex::MultiFab const* density_ratio,
    amrex::MultiFab const* attachment_frequency_cell,
    bool clear_sources)
{
    BL_PROFILE("LowEnergyFluidState::UpdateFromIonizationSource");
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::runtime_error("LowEnergyFluidState requires non-negative dt");
    }
    if (config.attachment_model != "constant"
        && config.attachment_model != "en_table_attachment_v1") {
        throw std::runtime_error(
            "rrea.fluid_attachment_model must be 'constant'"
            " or 'en_table_attachment_v1'");
    }
    bool const table_attachment = config.attachment_model == "en_table_attachment_v1";
    if (table_attachment != (attachment_frequency_cell != nullptr)) {
        throw std::runtime_error(
            "en_table_attachment_v1 requires exactly its evaluated "
            "attachment-frequency cell field (and no other model may bind one)");
    }

    double const attachment_frequency = std::max(config.attachment_frequency_s, 0.0);
    double const attachment_decay = std::exp(-attachment_frequency * dt);
    double const source_attachment_factor = (attachment_frequency > 0.0)
        ? (1.0 - attachment_decay) / attachment_frequency
        : dt;
    for (double coefficient : {
             config.electron_ion_recombination_coefficient_m3_per_s,
             config.ion_ion_recombination_coefficient_m3_per_s}) {
        if (!std::isfinite(coefficient) || coefficient < 0.0) {
            throw std::runtime_error(
                "fluid recombination coefficients must be finite and non-negative");
        }
    }
    double const electron_ion_recombination =
        config.electron_ion_recombination_coefficient_m3_per_s;
    double const ion_ion_recombination =
        config.ion_ion_recombination_coefficient_m3_per_s;
    double const detachment = std::max(config.detachment_frequency_s, 0.0);
    double const detachment_fraction =
        detachment > 0.0 ? -std::expm1(-detachment * dt) : 0.0;

    amrex::Array4<amrex::Real const> attach_factors_default{};

    for (amrex::MFIter mfi(*m_ne_low, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const ne = m_ne_low->array(mfi);
        auto const np = m_n_pos->array(mfi);
        auto const nn = m_n_neg->array(mfi);
        auto const src_low_electron = m_s_low_electron->array(mfi);
        auto const src_positive_ion = m_s_positive_ion->array(mfi);
        auto const src_direct_negative_ion = m_s_direct_negative_ion->array(mfi);
        auto const fluid_charge = m_rho_fluid->array(mfi);
        auto const expected_species_charge = m_charge_residual->array(mfi);
        auto const attach_nu_cell = table_attachment
            ? attachment_frequency_cell->const_array(mfi)
            : attach_factors_default;
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // The constant model keeps the precomputed scalar factors.
                // The en_table path reads the caller-evaluated nu(E/N, rho)
                // field, which changes every step with the field, and uses
                // expm1 so the source integral stays accurate at small nu*dt.
                double nu = attachment_frequency;
                double decay = attachment_decay;
                double source_factor = source_attachment_factor;
                if (table_attachment) {
                    nu = static_cast<double>(attach_nu_cell(i, j, k));
                    double const x = nu * dt;
                    decay = std::exp(-x);
                    source_factor = (nu > 0.0) ? -std::expm1(-x) / nu : dt;
                }
                double const low_electron_source_rate =
                    amrex::max(src_low_electron(i, j, k), 0.0);
                double const positive_ion_source_density =
                    amrex::max(src_positive_ion(i, j, k), 0.0) * dt;
                double const direct_negative_ion_source_density =
                    amrex::max(src_direct_negative_ion(i, j, k), 0.0) * dt;
                double const low_electron_source_density = low_electron_source_rate * dt;
                double const old_ne = amrex::max(ne(i, j, k), 0.0);
                double const old_np = amrex::max(np(i, j, k), 0.0);
                double const old_nn = amrex::max(nn(i, j, k), 0.0);

                // Attachment transfers one electron to one negative ion and
                // recombination removes an equal positive/negative pair.
                // Therefore only the explicit source channels change the
                // authoritative conservative fluid charge.  Update it from
                // its prior persisted value rather than reconstructing a
                // cancellation-prone difference of three large densities.
                fluid_charge(i, j, k) += qe * (
                    positive_ion_source_density
                    - direct_negative_ion_source_density
                    - low_electron_source_density);

                // Independently derive the species-space expected value so
                // the charge-conserving reactions can still be audited.
                expected_species_charge(i, j, k) = qe * (
                    old_np + positive_ion_source_density
                    - old_nn - direct_negative_ion_source_density
                    - old_ne - low_electron_source_density);

                // Exact update for dne/dt = S_low - nu_att*ne over this step,
                // assuming the ionization source is constant during dt.
                double const new_ne = (nu > 0.0)
                    ? old_ne * decay
                        + low_electron_source_rate * source_factor
                    : old_ne + low_electron_source_density;
                double const attached = amrex::max(
                    old_ne + low_electron_source_density - new_ne,
                    0.0);

                ne(i, j, k) = amrex::max(new_ne, 0.0);
                np(i, j, k) = old_np + positive_ion_source_density;
                nn(i, j, k) = old_nn
                    + direct_negative_ion_source_density + attached;

                // Detachment n_neg -> n_e (exact linear transfer, charge
                // conserving; np - nn - ne is invariant).
                if (detachment_fraction > 0.0) {
                    double const freed = nn(i, j, k) * detachment_fraction;
                    nn(i, j, k) -= freed;
                    ne(i, j, k) += freed;
                }

                // Effective O4+ electron-ion recombination, followed by the
                // separate positive/negative-ion channel.  Each substep is
                // exact for its governing pair ODE; the explicit order is the
                // project model and converges under the timestep ladder.
                double const electron_ion_loss = exact_pair_loss(
                    ne(i, j, k),
                    np(i, j, k),
                    electron_ion_recombination,
                    dt);
                ne(i, j, k) = amrex::max(
                    ne(i, j, k) - electron_ion_loss, 0.0);
                np(i, j, k) = amrex::max(
                    np(i, j, k) - electron_ion_loss, 0.0);

                double const ion_ion_loss = exact_pair_loss(
                    np(i, j, k),
                    nn(i, j, k),
                    ion_ion_recombination,
                    dt);
                np(i, j, k) = amrex::max(
                    np(i, j, k) - ion_ion_loss, 0.0);
                nn(i, j, k) = amrex::max(
                    nn(i, j, k) - ion_ion_loss, 0.0);

                if (clear_sources) {
                    src_low_electron(i, j, k) = 0.0;
                    src_positive_ion(i, j, k) = 0.0;
                    src_direct_negative_ion(i, j, k) = 0.0;
                }
            });
    }

    // Fail closed if any nominally charge-conserving attachment or
    // recombination path changes charge.  Normalize cell-by-cell by the
    // participating charge magnitudes so a neutral, high-density plasma does
    // not use a near-zero net-charge denominator.
    for (amrex::MFIter mfi(*m_ne_low, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const ne = m_ne_low->const_array(mfi);
        auto const np = m_n_pos->const_array(mfi);
        auto const nn = m_n_neg->const_array(mfi);
        auto const residual = m_charge_residual->array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                double const expected = residual(i, j, k);
                double const actual = qe * (
                    np(i, j, k) - nn(i, j, k) - ne(i, j, k));
                double const scale = std::max(
                    std::abs(expected)
                        + qe * (std::abs(np(i, j, k))
                            + std::abs(nn(i, j, k))
                            + std::abs(ne(i, j, k))),
                    std::numeric_limits<double>::min());
                residual(i, j, k) = std::abs(actual - expected) / scale;
            });
    }
    bool const reaction_residual_nonfinite =
        m_charge_residual->contains_nan(0, 1, 0, false);
    m_last_reaction_charge_residual = reaction_residual_nonfinite
        ? std::numeric_limits<double>::quiet_NaN()
        : static_cast<double>(m_charge_residual->norm0(0, 0, false));
    double const reaction_tolerance = 512.0
        * static_cast<double>(std::numeric_limits<amrex::Real>::epsilon());
    if (!std::isfinite(m_last_reaction_charge_residual)
        || m_last_reaction_charge_residual > reaction_tolerance) {
        std::ostringstream message;
        message << std::scientific << std::setprecision(17)
                << "low-energy attachment/recombination violated local charge "
                   "conservation: normalized="
                << m_last_reaction_charge_residual;
        throw std::runtime_error(message.str());
    }
    // m_rho_fluid was advanced directly by the explicit source charge above.
    // Do not replace it with the cancellation-prone species reconstruction.
}

double LowEnergyFluidState::MaxReactionFrequency(
    LowEnergyFluidConfig const& config,
    amrex::MultiFab const* attachment_frequency_cell)
{
    bool const table_attachment = config.attachment_model == "en_table_attachment_v1";
    if (table_attachment != (attachment_frequency_cell != nullptr)) {
        throw std::runtime_error(
            "reaction stiffness requires the selected attachment binding");
    }
    double const nu_constant = std::max(config.attachment_frequency_s, 0.0);
    double const detachment = std::max(config.detachment_frequency_s, 0.0);
    double const alpha_ei = config.electron_ion_recombination_coefficient_m3_per_s;
    double const alpha_ii = config.ion_ion_recombination_coefficient_m3_per_s;
    amrex::Array4<amrex::Real const> nu_default{};
    for (amrex::MFIter mfi(*m_ne_low, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const ne = m_ne_low->const_array(mfi);
        auto const np = m_n_pos->const_array(mfi);
        auto const nn = m_n_neg->const_array(mfi);
        auto const rate = m_charge_residual->array(mfi);
        auto const nu = table_attachment
            ? attachment_frequency_cell->const_array(mfi) : nu_default;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            double const attachment = table_attachment
                ? static_cast<double>(nu(i, j, k)) : nu_constant;
            rate(i, j, k) = attachment + detachment
                + alpha_ei * (amrex::max(ne(i, j, k), 0.0)
                              + amrex::max(np(i, j, k), 0.0))
                + alpha_ii * (amrex::max(np(i, j, k), 0.0)
                              + amrex::max(nn(i, j, k), 0.0));
        });
    }
    return static_cast<double>(m_charge_residual->norm0(0, 0, false));
}

void LowEnergyFluidState::ComputeConductivity(
    LowEnergyFluidConfig const& config,
    amrex::MultiFab const* density_ratio,
    amrex::MultiFab const* electron_mobility_cell)
{
    BL_PROFILE("LowEnergyFluidState::ComputeConductivity");
    // Only carriers that are actually advanced contribute to the implicit
    // conduction current.  Counting a disabled species in sigma while its
    // charge remains stationary violates charge continuity and creates a
    // memoryless, unphysical field sink.
    ValidateCarrierTransportConfig(config, false);
    bool const include_ions = config.ion_drift_enable;
    bool const include_electrons = config.electron_drift_enable;
    bool const electron_mobility_from_field = include_electrons
        && config.electron_mobility_model == "en_table_flux_v1";
    if (electron_mobility_from_field != (electron_mobility_cell != nullptr)) {
        throw std::runtime_error(
            "en_table_flux_v1 diagnostic conductivity requires exactly its "
            "evaluated electron-mobility cell field");
    }
    // Resolve the mobility law outside the per-cell loop.
    MobilityLaw const mob_pos = resolve_mobility_law(
        config.ion_mobility_model,
        config.positive_ion_mobility_m2_per_vs,
        config.positive_ion_reduced_mobility_stp_m2_per_vs,
        config.ion_mobility_density_ratio_floor);
    MobilityLaw const mob_neg = resolve_mobility_law(
        config.ion_mobility_model,
        config.negative_ion_mobility_m2_per_vs,
        config.negative_ion_reduced_mobility_stp_m2_per_vs,
        config.ion_mobility_density_ratio_floor);
    // The electron mobility entering sigma must match the one used by the
    // drift advection, or the implicit field relaxation and the persisted
    // conduction charge would disagree.
    // No reduced arm for electrons: "absolute" reads the constant and
    // "en_table_flux_v1" reads the caller-evaluated cell field, so the
    // reduced-mobility slot is never consulted.
    MobilityLaw const mob_ele = resolve_mobility_law(
        config.electron_mobility_model,
        config.electron_mobility_m2_per_vs,
        0.0,
        config.ion_mobility_density_ratio_floor);
    double const scalar_density_ratio = config.transport_density_ratio;
    for (amrex::MFIter mfi(*m_sigma, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const ne = m_ne_low->const_array(mfi);
        auto const np = m_n_pos->const_array(mfi);
        auto const nn = m_n_neg->const_array(mfi);
        auto const den = density_ratio ? density_ratio->const_array(mfi) : amrex::Array4<amrex::Real const>{};
        bool const has_density = density_ratio != nullptr;
        auto const mu_e_cell = electron_mobility_from_field
            ? electron_mobility_cell->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        auto const sigma = m_sigma->array(mfi);
        auto const tau = m_tau_m->array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                double const dens = checked_density_ratio(
                    den, has_density, i, j, k, scalar_density_ratio);
                double value = 0.0;
                if (include_electrons) {
                    double const mu_e_local = electron_mobility_from_field
                        ? static_cast<double>(mu_e_cell(i, j, k))
                        : mob_ele.At(dens);
                    value += mu_e_local * amrex::max(ne(i, j, k), 0.0);
                }
                if (include_ions) {
                    value += mob_pos.At(dens) * amrex::max(np(i, j, k), 0.0)
                        + mob_neg.At(dens) * amrex::max(nn(i, j, k), 0.0);
                }
                value *= qe;
                sigma(i, j, k) = amrex::max(value, 0.0);
                tau(i, j, k) = (sigma(i, j, k) > 0.0) ? eps0 / sigma(i, j, k) : 1.0e300;
            });
    }
}

void LowEnergyFluidState::ComputeFluidChargeDensity()
{
    BL_PROFILE("LowEnergyFluidState::ComputeFluidChargeDensity");
    for (amrex::MFIter mfi(*m_rho_fluid, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.tilebox();
        auto const ne = m_ne_low->const_array(mfi);
        auto const np = m_n_pos->const_array(mfi);
        auto const nn = m_n_neg->const_array(mfi);
        auto const rho = m_rho_fluid->array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                double const positive = amrex::max(np(i, j, k), 0.0);
                double const negative_ion = amrex::max(nn(i, j, k), 0.0);
                double const electron = amrex::max(ne(i, j, k), 0.0);
                double net = positive - negative_ion - electron;
                // Ionization deposits the
                // electron and positive-ion columns with identical weights,
                // so a quasi-neutral cell's net charge is pure floating-point
                // cancellation noise. A cell whose net charge
                // is below the round-off scale of its own carrier columns is
                // exactly neutral; real net charge always exceeds this bound.
                double const formation_scale =
                    positive + negative_ion + electron;
                if (amrex::Math::abs(net) <= 64.0
                        * std::numeric_limits<double>::epsilon()
                        * formation_scale) {
                    net = 0.0;
                }
                rho(i, j, k) = qe * net;
            });
    }
}

}  // namespace rrea
