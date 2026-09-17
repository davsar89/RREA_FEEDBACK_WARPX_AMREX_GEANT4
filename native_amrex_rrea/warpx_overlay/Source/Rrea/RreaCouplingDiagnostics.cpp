// The reduced-diagnostic CSV of RreaWarpXCoupling: the per-append scalar
// census of the fluid, event, ledger and field state.

#include "RreaWarpXCoupling.H"

#include "RreaCouplingDetail.H"
#include "RreaParticleSweep.H"

#include "rrea/RreaCsvTextUtil.H"
#include "rrea/RreaReducedObservables.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_Array4.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rrea::warpx {

using detail::checkpoint_reduced_diagnostic_schema_version;
using detail::collective_create_dir_all;
using detail::global_count_sum;
using detail::global_real_min;
using detail::global_vector_max;
using detail::global_vector_sum;
using detail::join_path;
using detail::photon_species_energy;

namespace {

std::string step_dir_name(int step)
{
    std::ostringstream stream;
    stream << "step" << std::setw(6) << std::setfill('0') << step;
    return stream.str();
}

using rrea::csv_text::split_csv_line;

amrex::Real sum_field(amrex::MultiFab const& mf)
{
    return mf.sum(0, false);
}

amrex::Real max_field(amrex::MultiFab const& mf)
{
    return mf.max(0, 0, false);
}

amrex::Real min_field(amrex::MultiFab const& mf)
{
    return mf.min(0, 0, false);
}

// Long companion to the detail header's batched real reductions.
void global_vector_long_sum(std::vector<amrex::Long>& values)
{
    if (!values.empty()) {
        if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            amrex::Abort("RREA vector reduction exceeds AMReX int count limit");
        }
        amrex::ParallelDescriptor::ReduceLongSum(
            values.data(),
            static_cast<int>(values.size()));
    }
}

amrex::Real volume_integral_rz(amrex::MultiFab const& mf, amrex::Geometry const& geom)
{
    amrex::Real local = amrex::Real(0.0);
    for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // Clip to the physical domain: the nodal-tagged coupled grids carry
        // one extra (inert) row beyond ProbHi that must not count as volume.
        auto const bx = mfi.validbox() & geom.Domain();
        if (!bx.ok()) {
            continue;
        }
        auto const arr = mf.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            local += arr(i, j, k) * cell_volume_rz(geom, i, j);
        });
    }
    amrex::ParallelDescriptor::ReduceRealSum(local);
    return local;
}

struct SpeciesWeightStatistics {
    amrex::Long macro_count = 0;
    amrex::Real total_weight = amrex::Real(0.0);
    amrex::Real sum_weight_squared = amrex::Real(0.0);
    amrex::Real max_weight = std::numeric_limits<amrex::Real>::quiet_NaN();

    [[nodiscard]] amrex::Real EffectiveSampleSize() const noexcept
    {
        return sum_weight_squared > amrex::Real(0.0)
            ? total_weight * total_weight / sum_weight_squared
            : amrex::Real(0.0);
    }
};

std::array<SpeciesWeightStatistics, 3> species_weight_statistics(
    std::array<std::string, 3> const& names)
{
    std::array<SpeciesWeightStatistics, 3> stats;
    std::vector<amrex::Long> counts(names.size(), 0);
    std::vector<amrex::Real> sums(2 * names.size(), amrex::Real(0.0));
    std::vector<amrex::Real> maxima(
        names.size(), -std::numeric_limits<amrex::Real>::infinity());
    for (std::size_t species_index = 0; species_index < names.size(); ++species_index) {
        if (names[species_index].empty()) {
            continue;
        }
        RreaForEachValidParticle(
            WarpX::GetInstance().GetPartContainer()
                .GetParticleContainerFromName(names[species_index]),
            [&, species_index](RreaLiveParticle const& particle) {
                amrex::Real const w = particle.weight;
                ++counts[species_index];
                sums[2 * species_index] += w;
                sums[2 * species_index + 1] += w * w;
                maxima[species_index] = amrex::max(maxima[species_index], w);
            });
    }
    global_vector_long_sum(counts);
    global_vector_sum(sums);
    global_vector_max(maxima);
    for (std::size_t i = 0; i < names.size(); ++i) {
        stats[i].macro_count = counts[i];
        stats[i].total_weight = sums[2 * i];
        stats[i].sum_weight_squared = sums[2 * i + 1];
        stats[i].max_weight = counts[i] > 0
            ? maxima[i]
            : std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    return stats;
}

}  // namespace

amrex::Real detail::maximum_represented_charge_e(
    std::array<std::string, 3> const& names)
{
    // One electron floors an empty domain: the gate needs a positive quantum
    // before any macro exists, and no smaller one is representable.
    amrex::Real maximum = amrex::Real(1.0);
    for (auto const& stats : species_weight_statistics(names)) {
        if (stats.macro_count > 0) {
            maximum = amrex::max(maximum, stats.max_weight);
        }
    }
    return maximum;
}

namespace {

amrex::Real electron_species_weight_above_energy(
    std::string const& name,
    amrex::Real threshold_eV)
{
    if (name.empty()) {
        return amrex::Real(0.0);
    }
    amrex::Real local = amrex::Real(0.0);
    RreaForEachValidParticle(
        WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(name),
        [&](RreaLiveParticle const& p) {
            if (p.KineticEnergyEv() >= threshold_eV) {
                local += p.weight;
            }
        });
    amrex::ParallelDescriptor::ReduceRealSum(local);
    return local;
}

// Signed vertical current moment of one
// kinetic species, M_z = q * sum_i w_i v_z,i [A*m] with v_z = u_z / gamma
// (u is the stored proper velocity).  threshold_eV > 0 restricts the sum to
// kinetic energies at or above the cutoff.  Recorded for the offline radio
// waveform reconstruction (architecture guide, "Radio-emission waveforms");
// diagnostics only -- never fed back into the physics.
amrex::Real species_current_moment_z_A_m(
    std::string const& name,
    amrex::Real charge_C,
    amrex::Real threshold_eV)
{
    if (name.empty()) {
        return amrex::Real(0.0);
    }
    amrex::Real local = amrex::Real(0.0);
    RreaForEachValidParticle(
        WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(name),
        [&](RreaLiveParticle const& p) {
            if (threshold_eV > amrex::Real(0.0)
                && p.KineticEnergyEv() < threshold_eV) {
                return;
            }
            local += p.weight * p.uz / p.Gamma();
        });
    amrex::ParallelDescriptor::ReduceRealSum(local);
    return charge_C * local;
}

constexpr int kRreaCurrentProfileBins = 128;

void species_current_profile_z(
    std::string const& name,
    amrex::Real charge_C,
    amrex::Real z_lo_m,
    amrex::Real inv_dz_bin,
    std::vector<amrex::Real>& bins)
{
    if (name.empty()) {
        return;
    }
    RreaForEachValidParticle(
        WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(name),
        [&](RreaLiveParticle const& p) {
            int bin = static_cast<int>((p.z - z_lo_m) * inv_dz_bin);
            bin = amrex::min(amrex::max(bin, 0), kRreaCurrentProfileBins - 1);
            bins[static_cast<std::size_t>(bin)] +=
                charge_C * p.weight * p.uz / p.Gamma();
        });
}

std::vector<amrex::Real> fluid_current_profile_z(
    amrex::MultiFab const& ez,
    amrex::MultiFab const& sigma,
    amrex::Geometry const& geom)
{
    std::vector<amrex::Real> bins(kRreaCurrentProfileBins, amrex::Real(0.0));
#if (AMREX_SPACEDIM >= 2)
    auto const dx = geom.CellSizeArray();
    auto const plo = cell_index_origin(geom);
    amrex::Real const z_lo = geom.ProbLo(1);
    amrex::Real const inv_dz_bin =
        static_cast<amrex::Real>(kRreaCurrentProfileBins)
        / (geom.ProbHi(1) - geom.ProbLo(1));
    for (amrex::MFIter mfi(ez, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const bx = mfi.validbox();
        auto const ez_arr = ez.const_array(mfi);
        auto const sigma_arr = sigma.const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) noexcept {
            amrex::Real const z = plo[1]
                + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1];
            amrex::Real const sigma_value =
                amrex::max(sigma_arr(i, j, k), amrex::Real(0.0));
            int bin = static_cast<int>((z - z_lo) * inv_dz_bin);
            bin = amrex::min(amrex::max(bin, 0), kRreaCurrentProfileBins - 1);
            bins[static_cast<std::size_t>(bin)] +=
                sigma_value * ez_arr(i, j, k) * cell_volume_rz(geom, i, j);
        });
    }
#else
    amrex::ignore_unused(ez, sigma, geom);
#endif
    return bins;
}

}  // namespace

// Append-only CSV streams: a header that differs from the current schema
// means the engine changed (an epoch deviation).  The old stream is archived
// as <stem>_pre_<step>.csv and a fresh one started, never migrated by hand.
// Returns true when the caller must write the header (absent or archived).
bool RreaWarpXCoupling::ArchiveCsvIfHeaderChanged(
    std::string const& path, std::string const& header, int step) const
{
    std::string existing_header;
    {
        std::ifstream existing(path);
        if (!existing || !std::getline(existing, existing_header)) {
            return true;
        }
    }
    if (!existing_header.empty() && existing_header.back() == '\r') {
        existing_header.pop_back();
    }
    if (existing_header == header) {
        return false;
    }
    std::string const archived = path.substr(0, path.size() - 4)
        + "_pre_" + std::to_string(step) + ".csv";
    ToleratedRestartEpochDeviationOrAbort(
        "RREA CSV schema changed; archived " + path + " as " + archived);
    if (std::rename(path.c_str(), archived.c_str()) != 0) {
        amrex::Abort("RREA could not archive " + path);
    }
    return true;
}

void RreaWarpXCoupling::AppendReducedDiagnostics(
    std::string const& dir,
    int step,
    amrex::Real time_s) const
{
    if (!m_initialized || !m_advance || !m_events) {
        return;
    }
    auto const& fluid = m_advance->Fluid(0);
    auto const& geom0 = m_level_bindings[0].geom;
    amrex::Real const total_ne_low = sum_field(fluid.LowElectronDensity());
    amrex::Real const total_n_pos = sum_field(fluid.PositiveIonDensity());
    amrex::Real const total_n_neg = sum_field(fluid.NegativeIonDensity());
    amrex::Real const total_rho_fluid = sum_field(fluid.FluidChargeDensity());
    amrex::Real const n_low_e_real =
        volume_integral_rz(fluid.LowElectronDensity(), geom0);
    amrex::Real const n_pos_real =
        volume_integral_rz(fluid.PositiveIonDensity(), geom0);
    amrex::Real const n_neg_real =
        volume_integral_rz(fluid.NegativeIonDensity(), geom0);
    amrex::Real const rho_fluid_c =
        volume_integral_rz(fluid.FluidChargeDensity(), geom0);
    auto const particle_weight_stats = species_weight_statistics(
        {m_seed_species_name, m_photon_species_name, m_positron_species_name});
    amrex::Real const n_kinetic_e_real = particle_weight_stats[0].total_weight;
    amrex::Real const n_kinetic_e_ge_1mev_real =
        electron_species_weight_above_energy(m_seed_species_name, amrex::Real(1.0e6));
    // Near-cutoff band census.  Every "low" bucket in this file is defined
    // relative to the selected transport cutoff, so weight sitting just above
    // it is invisible inside the undifferentiated kinetic bucket. The 100-keV
    // probe is near the runaway-energy scale for the profiled fields, so its
    // complement is the weight
    // that is transported but NOT runaway-capable, which is what a rise in
    // the cutoff removes first.  A probe at or below the cutoff would return
    // the species total and measure nothing; keeping the energy in the column
    // name rather than a deck knob means changing it creates a distinct column.
    amrex::Real const n_kinetic_e_ge_100kev_real =
        electron_species_weight_above_energy(m_seed_species_name, amrex::Real(1.0e5));
    amrex::Real const n_positron_ge_100kev_real =
        electron_species_weight_above_energy(m_positron_species_name, amrex::Real(1.0e5));
    amrex::Real const n_photon_real = particle_weight_stats[1].total_weight;
    amrex::Real const n_positron_real = particle_weight_stats[2].total_weight;
    amrex::Real const n_inst_e_real = n_kinetic_e_real + n_low_e_real;
    // Gather all per-append scalar counters into
    // two array allreduces (long-sum, real-sum) instead of sequential scalar
    // allreduces.  Elementwise arithmetic is unchanged; the unpack order below
    // matches the initializer order exactly.
    std::vector<amrex::Long> batched_counts = {
        static_cast<amrex::Long>(m_kinetic_secondary_electron_count),
        static_cast<amrex::Long>(m_photon_created_count),
        static_cast<amrex::Long>(m_ledger.photon_interaction_count),
        static_cast<amrex::Long>(m_ledger.electron_brems_photon_count),
        static_cast<amrex::Long>(m_positron_created_count),
        static_cast<amrex::Long>(m_ledger.compton_interaction_count),
        static_cast<amrex::Long>(m_ledger.photoelectric_interaction_count),
        static_cast<amrex::Long>(m_ledger.pair_production_count),
        static_cast<amrex::Long>(m_ledger.pair_nuclear_count),
        static_cast<amrex::Long>(m_ledger.pair_triplet_count),
        static_cast<amrex::Long>(m_ledger.positron_thermalized_count),
        static_cast<amrex::Long>(m_ledger.positron_bhabha_event_count),
        static_cast<amrex::Long>(m_ledger.positron_brems_event_count),
        static_cast<amrex::Long>(m_positron_annihilation_count),
        static_cast<amrex::Long>(m_positron_annihilation_at_rest_count),
        static_cast<amrex::Long>(m_positron_annihilation_in_flight_count),
        static_cast<amrex::Long>(m_ledger.hard_moller_event_count)};
    global_vector_long_sum(batched_counts);
    std::size_t batched_count_index = 0;
    auto next_batched_count = [&]() {
        return static_cast<std::uint64_t>(batched_counts[batched_count_index++]);
    };
    std::vector<amrex::Real> batched_sums = {
        m_kinetic_secondary_weight,
        m_photon_created_weight,
        m_ledger.photon_interaction_weight,
        m_ledger.total_brems_photon_energy_eV,
        m_ledger.photon_continuous_soft_brems_energy_eV,
        m_ledger.transport_ionization_weight,
        m_ion_pair_source_weight,
        m_ion_pair_source_z_weighted_m,
        m_ledger.transport_low_secondary_weight,
        m_ledger.transport_kinetic_secondary_weight,
        m_ledger.transport_energy_loss_eV,
        m_positron_created_weight,
        m_ledger.compton_interaction_weight,
        m_ledger.photoelectric_interaction_weight,
        m_ledger.pair_production_weight,
        m_ledger.pair_nuclear_weight,
        m_ledger.pair_triplet_weight,
        m_ledger.pair_recoil_electron_energy_eV,
        m_ledger.pair_local_deposit_energy_eV,
        m_ledger.photon_seeded_electron_weight,
        m_ledger.photon_seeded_positron_weight,
        m_ledger.photon_seeded_kinetic_electron_weight,
        m_ledger.photon_seeded_low_electron_weight,
        m_ledger.photon_seeded_kinetic_positron_weight,
        m_ledger.photon_seeded_below_cutoff_positron_weight,
        m_ledger.photon_absorbed_energy_eV,
        m_ledger.positron_transport_energy_loss_eV,
        m_positron_annihilation_photon_energy_eV};
    global_vector_sum(batched_sums);
    std::size_t batched_sum_index = 0;
    auto next_batched_sum = [&]() {
        return batched_sums[batched_sum_index++];
    };
    std::uint64_t const global_kinetic_secondary_electron_count = next_batched_count();
    std::uint64_t const global_photon_created_count = next_batched_count();
    std::uint64_t const global_photon_interaction_count = next_batched_count();
    std::uint64_t const global_electron_brems_photon_count = next_batched_count();
    std::uint64_t const global_positron_created_count = next_batched_count();
    std::uint64_t const global_compton_interaction_count = next_batched_count();
    std::uint64_t const global_photoelectric_interaction_count = next_batched_count();
    std::uint64_t const global_pair_production_count = next_batched_count();
    std::uint64_t const global_pair_nuclear_count = next_batched_count();
    std::uint64_t const global_pair_triplet_count = next_batched_count();
    std::uint64_t const global_positron_thermalized_count = next_batched_count();
    std::uint64_t const global_positron_bhabha_event_count = next_batched_count();
    std::uint64_t const global_positron_brems_event_count = next_batched_count();
    std::uint64_t const global_positron_annihilation_count = next_batched_count();
    std::uint64_t const global_positron_annihilation_at_rest_count = next_batched_count();
    std::uint64_t const global_positron_annihilation_in_flight_count = next_batched_count();
    std::uint64_t const global_hard_moller_event_count = next_batched_count();
    amrex::Real const global_kinetic_secondary_weight = next_batched_sum();
    amrex::Real const global_photon_created_weight = next_batched_sum();
    amrex::Real const global_photon_interaction_weight = next_batched_sum();
    amrex::Real const global_total_brems_photon_energy_eV = next_batched_sum();
    amrex::Real const global_photon_continuous_soft_brems_energy_eV = next_batched_sum();
    amrex::Real const global_transport_ionization_weight = next_batched_sum();
    amrex::Real const global_ion_pair_source_weight = next_batched_sum();
    amrex::Real const global_ion_pair_source_z_weighted_m = next_batched_sum();
    amrex::Real const global_transport_low_secondary_weight = next_batched_sum();
    amrex::Real const global_transport_kinetic_secondary_weight = next_batched_sum();
    amrex::Real const global_transport_energy_loss_eV = next_batched_sum();
    amrex::Real const global_positron_created_weight = next_batched_sum();
    amrex::Real const global_compton_interaction_weight = next_batched_sum();
    amrex::Real const global_photoelectric_interaction_weight = next_batched_sum();
    amrex::Real const global_pair_production_weight = next_batched_sum();
    amrex::Real const global_pair_nuclear_weight = next_batched_sum();
    amrex::Real const global_pair_triplet_weight = next_batched_sum();
    amrex::Real const global_pair_recoil_electron_energy_eV = next_batched_sum();
    amrex::Real const global_pair_local_deposit_energy_eV = next_batched_sum();
    amrex::Real const global_photon_seeded_electron_weight = next_batched_sum();
    amrex::Real const global_photon_seeded_positron_weight = next_batched_sum();
    amrex::Real const global_photon_seeded_kinetic_electron_weight = next_batched_sum();
    amrex::Real const global_photon_seeded_low_electron_weight = next_batched_sum();
    amrex::Real const global_photon_seeded_kinetic_positron_weight = next_batched_sum();
    amrex::Real const global_photon_seeded_below_cutoff_positron_weight = next_batched_sum();
    amrex::Real const global_photon_absorbed_energy_eV = next_batched_sum();
    amrex::Real const global_positron_transport_energy_loss_eV = next_batched_sum();
    amrex::Real const global_positron_annihilation_photon_energy_eV = next_batched_sum();
    if (batched_count_index != batched_counts.size()
        || batched_sum_index != batched_sums.size()) {
        amrex::Abort("RREA reduced-diagnostic registry index mismatch");
    }
    amrex::Real const n_cumulative_e_created =
        m_injected_physical_weight
        + global_transport_low_secondary_weight
        + global_transport_kinetic_secondary_weight;
    amrex::Real const max_ne_low = max_field(fluid.LowElectronDensity());
    amrex::Real const max_sigma = max_field(fluid.Conductivity());
    amrex::Real const min_tau_m = min_field(fluid.MaxwellTime());
    auto const& field_init = m_advance_config.field_initializer;
    amrex::Real const e0_abs = std::abs(field_init.ez_acceleration_v_per_m);
    // The field_fraction_*_accel_region columns cover only where the ambient
    // field is imposed: the profile-active band when an altitude profile is
    // loaded (else z <= acceleration_z_max_m), inside the taper core.  The
    // taper fade and the profile's vertical nulls are ~0-ambient geometry from
    // t=0, so including them floors min/p10 and reads as instant screening.
    bool const profile_band_valid =
        std::isfinite(static_cast<double>(m_profile_active_z_min_m))
        && std::isfinite(static_cast<double>(m_profile_active_z_max_m))
        && m_profile_active_z_max_m > m_profile_active_z_min_m;
    auto const field_metrics = acceleration_field_metrics_rz(
        *m_level_bindings[0].efield_z,
        geom0,
        e0_abs,
        profile_band_valid ? m_profile_active_z_min_m : amrex::Real(0.0),
        profile_band_valid ? m_profile_active_z_max_m
                           : amrex::Real(field_init.acceleration_z_max_m),
        field_init.TaperActive()
            ? amrex::Real(field_init.taper_r_start_m)
            : std::numeric_limits<amrex::Real>::infinity());
    // Realized E/N over the same acceleration region as the field-fraction
    // columns; always on so the production closure band stays visible.
    auto const en_metrics = en_td_region_metrics_rz(
        *m_level_bindings[0].efield_x,
        *m_level_bindings[0].efield_z,
        m_level_bindings[0].density_ratio,
        amrex::Real(m_advance_config.fluid.transport_density_ratio),
        fluid.LowElectronDensity(),
        m_events->LowElectronSourceRate(0),
        geom0,
        profile_band_valid ? m_profile_active_z_min_m : amrex::Real(0.0),
        profile_band_valid ? m_profile_active_z_max_m
                           : amrex::Real(field_init.acceleration_z_max_m),
        field_init.TaperActive()
            ? amrex::Real(field_init.taper_r_start_m)
            : std::numeric_limits<amrex::Real>::infinity());
    static constexpr std::array<amrex::Real, 5> channel_radii_m = {
        amrex::Real(42.0),
        amrex::Real(50.0),
        amrex::Real(100.0),
        amrex::Real(200.0),
        amrex::Real(1200.0)};
    static constexpr std::array<char const*, 5> channel_labels = {
        "r42m",
        "r50m",
        "r100m",
        "r200m",
        "r1200m"};
    std::vector<std::pair<std::string, amrex::Real>> extra_metrics;
    extra_metrics.reserve(64);
    // Global vertical current moments [A*m]
    // for offline radio-waveform reconstruction.  The fluid conduction
    // moment is integral sigma*Ez dV over the whole domain (sigma already
    // sums the low-energy electron and both ion drift channels); the kinetic
    // moments are exact per-macro sums q*w*v_z over every energy, so the
    // >=1 MeV column directly validates the offline -e*N*<beta_z>*c
    // approximation.  Diagnostics only; nothing feeds back into the physics.
    std::vector<amrex::Real> current_profile_fluid;
    std::vector<amrex::Real> current_profile_kinetic_e;
    std::vector<amrex::Real> current_profile_positron;
    amrex::Real current_profile_z_lo = amrex::Real(0.0);
    amrex::Real current_profile_z_hi = amrex::Real(0.0);
    {
        static constexpr amrex::Real elementary_charge_C = amrex::Real(rrea::qe);
        auto const global_ohmic = ohmic_discharge_metrics_rz(
            *m_level_bindings[0].efield_z,
            fluid.Conductivity(),
            geom0,
            geom0.ProbLo(1),
            geom0.ProbHi(1),
            std::numeric_limits<amrex::Real>::max());
        extra_metrics.emplace_back(
            "global_jz_fluid_A_m",
            global_ohmic.mean_sigma_ez_A_per_m2 * global_ohmic.volume_m3);
        extra_metrics.emplace_back(
            "fluid_current_rms_r_m", global_ohmic.fluid_current_rms_r_m);
        extra_metrics.emplace_back(
            "fluid_current_rms_z_m", global_ohmic.fluid_current_rms_z_m);
        extra_metrics.emplace_back(
            "global_jz_kinetic_e_A_m",
            species_current_moment_z_A_m(
                m_seed_species_name, -elementary_charge_C, amrex::Real(0.0)));
        extra_metrics.emplace_back(
            "global_jz_kinetic_e_ge_1MeV_A_m",
            species_current_moment_z_A_m(
                m_seed_species_name, -elementary_charge_C, amrex::Real(1.0e6)));
        extra_metrics.emplace_back(
            "global_jz_positron_A_m",
            species_current_moment_z_A_m(
                m_positron_species_name, elementary_charge_C, amrex::Real(0.0)));
        // Z-binned partial sums of the same moments, with one batched allreduce
        // for all three components.
        current_profile_z_lo = geom0.ProbLo(1);
        current_profile_z_hi = geom0.ProbHi(1);
        amrex::Real const inv_dz_bin =
            static_cast<amrex::Real>(kRreaCurrentProfileBins)
            / (current_profile_z_hi - current_profile_z_lo);
        current_profile_fluid = fluid_current_profile_z(
            *m_level_bindings[0].efield_z, fluid.Conductivity(), geom0);
        current_profile_kinetic_e.assign(kRreaCurrentProfileBins, amrex::Real(0.0));
        current_profile_positron.assign(kRreaCurrentProfileBins, amrex::Real(0.0));
        species_current_profile_z(
            m_seed_species_name, -elementary_charge_C,
            current_profile_z_lo, inv_dz_bin, current_profile_kinetic_e);
        species_current_profile_z(
            m_positron_species_name, elementary_charge_C,
            current_profile_z_lo, inv_dz_bin, current_profile_positron);
        std::vector<amrex::Real> all_bins;
        all_bins.reserve(3 * kRreaCurrentProfileBins);
        for (auto const* vec : {&current_profile_fluid,
                                &current_profile_kinetic_e,
                                &current_profile_positron}) {
            all_bins.insert(all_bins.end(), vec->begin(), vec->end());
        }
        amrex::ParallelDescriptor::ReduceRealSum(
            all_bins.data(), static_cast<int>(all_bins.size()));
        std::copy(all_bins.begin(),
                  all_bins.begin() + kRreaCurrentProfileBins,
                  current_profile_fluid.begin());
        std::copy(all_bins.begin() + kRreaCurrentProfileBins,
                  all_bins.begin() + 2 * kRreaCurrentProfileBins,
                  current_profile_kinetic_e.begin());
        std::copy(all_bins.begin() + 2 * kRreaCurrentProfileBins,
                  all_bins.end(),
                  current_profile_positron.begin());
    }
    amrex::Real const photon_alive_energy_eV = photon_species_energy(m_photon_species_name);
    std::vector<amrex::Real> photon_ledger_flows = {
        // External photon injection is absent; keep its zero flow explicit.
        amrex::Real(0.0),
        m_ledger.photon_escaped_energy_eV,
        m_ledger.photon_to_charged_energy_eV,
        m_ledger.photon_local_deposit_energy_eV};
    global_vector_sum(photon_ledger_flows);
    RreaPhotonLedgerFlows const photon_ledger{
        m_initial_photon_alive_energy_eV,
        photon_ledger_flows[0],
        global_total_brems_photon_energy_eV,
        global_positron_annihilation_photon_energy_eV,
        photon_alive_energy_eV,
        photon_ledger_flows[1],
        photon_ledger_flows[2],
        photon_ledger_flows[3]};
    amrex::Real const photon_ledger_left_eV = photon_ledger.SourceEv();
    amrex::Real const photon_ledger_right_eV = photon_ledger.SinkEv();
    amrex::Real const photon_ledger_residual_eV = photon_ledger.ResidualEv();
    amrex::Real const photon_ledger_relative_residual =
        photon_ledger.RelativeResidual();
    constexpr amrex::Real kPhotonLedgerRelativeResidualMax = amrex::Real(1.0e-8);
    if (!std::isfinite(static_cast<double>(photon_ledger_relative_residual))
        || photon_ledger_relative_residual > kPhotonLedgerRelativeResidualMax) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "RREA photon transport ledger failed "
                << "the collective relative-residual gate at step=" << step
                << ", time_s=" << time_s
                << ": relative_residual=" << photon_ledger_relative_residual
                << ", limit=" << kPhotonLedgerRelativeResidualMax
                << ", residual_eV=" << photon_ledger_residual_eV
                << ", source_throughput_eV=" << photon_ledger_left_eV
                << ", sink_throughput_eV=" << photon_ledger_right_eV
                << ", initial_alive_eV=" << m_initial_photon_alive_energy_eV
                << ", external_injected_eV=" << photon_ledger_flows[0]
                << ", brems_created_eV=" << global_total_brems_photon_energy_eV
                << ", annihilation_created_eV="
                << global_positron_annihilation_photon_energy_eV
                << ", alive_eV=" << photon_alive_energy_eV
                << ", escaped_eV=" << photon_ledger_flows[1]
                << ", to_charged_eV=" << photon_ledger_flows[2]
                << ", local_deposit_eV=" << photon_ledger_flows[3];
        amrex::Abort(message.str());
    }
    extra_metrics.emplace_back(
        "photon_continuous_soft_brems_energy_eV",
        global_photon_continuous_soft_brems_energy_eV);
    // profile_active ne_low-weighted |E|/E0_peak per channel radius: the
    // optical-emissions pipeline reads the largest recorded radius as its
    // field-screening input.
    if (profile_band_valid) {
        for (std::size_t index = 0; index < channel_radii_m.size(); ++index) {
            extra_metrics.emplace_back(
                std::string("profile_active_abs_e_over_e0_peak_ne_low_weighted_")
                    + channel_labels[index],
                weighted_field_fraction_rz(
                    *m_level_bindings[0].efield_z,
                    fluid.LowElectronDensity(),
                    geom0,
                    e0_abs,
                    m_profile_active_z_min_m,
                    m_profile_active_z_max_m,
                    channel_radii_m[index]).fraction);
        }
    }
    // Realized reduced-field columns (see en_td_region_metrics_rz): three
    // weightings plus the de Urquijo certified-range occupation fractions.
    extra_metrics.emplace_back("en_td_accel_min", en_metrics.min_en_td);
    extra_metrics.emplace_back("en_td_accel_max", en_metrics.max_en_td);
    {
        static constexpr std::array<char const*, 3> channel_names = {
            "vol", "ne_low", "src"};
        static constexpr std::array<char const*, 3> quantile_names = {
            "p10", "p50", "p90"};
        for (std::size_t channel = 0; channel < channel_names.size();
             ++channel) {
            extra_metrics.emplace_back(
                std::string("en_td_accel_") + channel_names[channel]
                    + "_weighted_mean",
                en_metrics.weighted_mean_en_td[channel]);
            for (std::size_t q = 0; q < quantile_names.size(); ++q) {
                extra_metrics.emplace_back(
                    std::string("en_td_accel_") + channel_names[channel]
                        + "_" + quantile_names[q],
                    en_metrics.p10_p50_p90[channel][q]);
            }
        }
    }
    extra_metrics.emplace_back(
        "en_td_accel_vol_frac_below_3td", en_metrics.vol_frac_below_3td);
    extra_metrics.emplace_back(
        "en_td_accel_vol_frac_above_30td", en_metrics.vol_frac_above_30td);
    extra_metrics.emplace_back(
        "en_td_accel_ne_low_frac_below_3td", en_metrics.ne_frac_below_3td);
    if (m_advance_config.electron_closure != nullptr
        && m_advance_config.electron_closure->Loaded()) {
        // Whole-domain realized range and held-cell count from the en_table
        // evaluations the step actually consumed (see
        // RreaAmrexAdvanceDiagnostics); above-range/invalid states abort, so
        // their absence here is load-bearing, not optimistic.
        auto const& advance_diag = m_advance->LastDiagnostics();
        extra_metrics.emplace_back(
            "closure_realized_min_en_td",
            advance_diag.closure_evaluated
                ? static_cast<amrex::Real>(advance_diag.closure_min_en_td)
                : std::numeric_limits<amrex::Real>::quiet_NaN());
        extra_metrics.emplace_back(
            "closure_realized_max_en_td",
            advance_diag.closure_evaluated
                ? static_cast<amrex::Real>(advance_diag.closure_max_en_td)
                : std::numeric_limits<amrex::Real>::quiet_NaN());
        extra_metrics.emplace_back(
            "closure_held_below_cells",
            static_cast<amrex::Real>(
                advance_diag.closure_held_below_cells));
        extra_metrics.emplace_back(
            "closure_held_below_electron_fraction",
            static_cast<amrex::Real>(
                advance_diag.closure_max_held_below_electron_fraction));
        extra_metrics.emplace_back(
            "closure_held_below_source_fraction",
            static_cast<amrex::Real>(
                advance_diag.closure_max_held_below_source_fraction));
        extra_metrics.emplace_back(
            "closure_held_below_conductivity_fraction",
            static_cast<amrex::Real>(
                advance_diag.closure_max_held_below_conductivity_fraction));
    }

    amrex::Real const total_s_low_electron = sum_field(m_events->LowElectronSourceRate(0));
    amrex::Real const total_s_positive_ion = sum_field(m_events->PositiveIonSourceRate(0));
    amrex::Real const total_s_direct_negative_ion =
        sum_field(m_events->DirectNegativeIonSourceRate(0));
    amrex::Real const ionization_event_count = sum_field(m_events->IonizationEventCount(0));
    amrex::Real const total_energy_loss_ev_m3 =
        sum_field(m_events->EnergyLossDensityEvPerM3(0));
    if (m_population_ceiling_policy == "adaptive_resample_v1") {
        // adaptive_resample_v1 population-control ledger (global cumulative).
        // "killed" records every negative weight increment (including partial
        // reductions), while "boost" records every positive increment.  The
        // CIC null-space move balances their total to deposition round-off in
        // each realization; orphan is structurally zero.
        std::vector<amrex::Real> resample_reals = {
            m_resample_killed_weight,
            m_resample_boost_weight,
            m_resample_orphan_weight,
            m_resample_split_weight,
            m_resample_killed_energy_eV,
            m_resample_boost_energy_eV,
            static_cast<amrex::Real>(m_resample_killed_count),
            static_cast<amrex::Real>(m_resample_split_count)};
        global_vector_sum(resample_reals);
        extra_metrics.emplace_back("resample_killed_weight", resample_reals[0]);
        extra_metrics.emplace_back("resample_boost_weight", resample_reals[1]);
        extra_metrics.emplace_back("resample_orphan_weight", resample_reals[2]);
        extra_metrics.emplace_back("resample_split_weight", resample_reals[3]);
        extra_metrics.emplace_back(
            "resample_killed_energy_eV", resample_reals[4]);
        extra_metrics.emplace_back(
            "resample_boost_energy_eV", resample_reals[5]);
        extra_metrics.emplace_back("resample_killed_count", resample_reals[6]);
        extra_metrics.emplace_back("resample_split_count", resample_reals[7]);
        extra_metrics.emplace_back(
            "resample_thin_rounds",
            static_cast<amrex::Real>(m_resample_thin_rounds));
        extra_metrics.emplace_back(
            "resample_electron_exact_capacity",
            static_cast<amrex::Real>(m_resample_exact_capacity[0]));
        extra_metrics.emplace_back(
            "resample_electron_geometric_floor",
            static_cast<amrex::Real>(m_resample_geometric_floor[0]));
        extra_metrics.emplace_back(
            "resample_positron_exact_capacity",
            static_cast<amrex::Real>(m_resample_exact_capacity[1]));
        extra_metrics.emplace_back(
            "resample_positron_geometric_floor",
            static_cast<amrex::Real>(m_resample_geometric_floor[1]));
    }
    static constexpr std::array<char const*, 3> particle_labels = {
        "kinetic_e", "photon", "positron"};
    for (std::size_t i = 0; i < particle_weight_stats.size(); ++i) {
        extra_metrics.emplace_back(
            std::string(particle_labels[i]) + "_N_eff",
            particle_weight_stats[i].EffectiveSampleSize());
        extra_metrics.emplace_back(
            std::string(particle_labels[i]) + "_max_weight",
            particle_weight_stats[i].max_weight);
    }
    extra_metrics.emplace_back(
        "profile_e0_peak_v_per_m",
        std::abs(m_advance_config.field_initializer.ez_acceleration_v_per_m));
    extra_metrics.emplace_back(
        "positive_ion_center_z_m",
        fluid.LastCarrierDriftDiagnostics().positive_ion_center_z_m);
    extra_metrics.emplace_back(
        "ion_pair_source_weight_cumulative", global_ion_pair_source_weight);
    extra_metrics.emplace_back(
        "ion_pair_source_z_weighted_m_cumulative",
        global_ion_pair_source_z_weighted_m);
    extra_metrics.emplace_back(
        "hard_moller_event_count",
        static_cast<amrex::Real>(global_hard_moller_event_count));
    extra_metrics.emplace_back(
        "electron_brems_photon_count",
        static_cast<amrex::Real>(global_electron_brems_photon_count));
    extra_metrics.emplace_back(
        "N_kinetic_e_ge_100keV_real", n_kinetic_e_ge_100kev_real);
    extra_metrics.emplace_back(
        "N_positron_ge_100keV_real", n_positron_ge_100kev_real);
    extra_metrics.emplace_back(
        "photon_seeded_kinetic_electron_weight",
        global_photon_seeded_kinetic_electron_weight);
    extra_metrics.emplace_back(
        "photon_seeded_low_electron_weight",
        global_photon_seeded_low_electron_weight);
    extra_metrics.emplace_back(
        "photon_seeded_kinetic_positron_weight",
        global_photon_seeded_kinetic_positron_weight);
    extra_metrics.emplace_back(
        "photon_seeded_below_cutoff_positron_weight",
        global_photon_seeded_below_cutoff_positron_weight);
    extra_metrics.emplace_back(
        "positron_bhabha_event_count",
        static_cast<amrex::Real>(global_positron_bhabha_event_count));
    extra_metrics.emplace_back(
        "positron_brems_event_count",
        static_cast<amrex::Real>(global_positron_brems_event_count));
    extra_metrics.emplace_back(
        "positron_annihilation_count",
        static_cast<amrex::Real>(global_positron_annihilation_count));
    extra_metrics.emplace_back(
        "positron_annihilation_at_rest_count",
        static_cast<amrex::Real>(global_positron_annihilation_at_rest_count));
    extra_metrics.emplace_back(
        "positron_annihilation_in_flight_count",
        static_cast<amrex::Real>(global_positron_annihilation_in_flight_count));
    // Net-chord current diagnostic, using values from this step's transport
    // (the scratch is rearmed every step).  The channel sums are the signed
    // charge (units e) that entered or left the kinetic population, made
    // visible rather than silently absorbed.  The segment-vs-endpoint
    // continuity identity is owned by rrea_segment_deposition_smoke and, in
    // the run, by the engine's material-continuity gate.
    {
        std::vector<amrex::Real> netchord_sums = {
            static_cast<amrex::Real>(m_netchord_segment_count),
            m_netchord_z_displacement_e_m,
            m_netchord_created_charge_e,
            m_netchord_escaped_charge_e,
            m_netchord_demoted_charge_e,
            m_netchord_annihilated_charge_e};
        global_vector_sum(netchord_sums);
        static constexpr amrex::Real qe_C = amrex::Real(rrea::qe);
        extra_metrics.emplace_back("netchord_segment_count", netchord_sums[0]);
        extra_metrics.emplace_back(
            "netchord_jz_moment_A_m",
            m_netchord_dt_s > amrex::Real(0.0)
                ? qe_C * netchord_sums[1] / m_netchord_dt_s
                : amrex::Real(0.0));
        extra_metrics.emplace_back(
            "netchord_created_charge_e", netchord_sums[2]);
        extra_metrics.emplace_back(
            "netchord_escaped_charge_e", netchord_sums[3]);
        extra_metrics.emplace_back(
            "netchord_demoted_charge_e", netchord_sums[4]);
        extra_metrics.emplace_back(
            "netchord_annihilated_charge_e", netchord_sums[5]);
    }
    // Maxwell TM columns: the eps0-Gauss monitor (roulette + boundary-escape
    // channels), the peak Btheta available to near-to-far analysis, the
    // scattered field energy, and the physical total-field energy change
    // relative to the ambient (which includes the ambient/scattered cross
    // term).
    {
        auto const& advance_diag = m_advance->LastDiagnostics();
        extra_metrics.emplace_back(
            "em_gauss_residual_rel",
            static_cast<amrex::Real>(
                advance_diag.maxwell_gauss_residual_normalized));
        extra_metrics.emplace_back(
            "em_charge_continuity_max_rel",
            static_cast<amrex::Real>(
                advance_diag.maxwell_charge_continuity_max_relative));
        extra_metrics.emplace_back(
            "em_charge_continuity_max_abs_e_per_cell",
            static_cast<amrex::Real>(
                advance_diag.maxwell_charge_continuity_max_abs_e_per_cell));
        extra_metrics.emplace_back(
            "em_charge_continuity_signed_accum_e",
            static_cast<amrex::Real>(
                advance_diag.maxwell_charge_continuity_signed_accum_e));
        extra_metrics.emplace_back(
            "em_btheta_max_T",
            static_cast<amrex::Real>(advance_diag.maxwell_btheta_max_t));
        extra_metrics.emplace_back(
            "em_max_dt_sigma_face_over_eps0",
            static_cast<amrex::Real>(
                advance_diag.maxwell_max_dt_sigma_face_over_eps0));
        extra_metrics.emplace_back(
            "fluid_explicit_diffusion_number",
            static_cast<amrex::Real>(
                advance_diag.fluid_max_explicit_diffusion_number));
        extra_metrics.emplace_back(
            "material_substeps",
            static_cast<amrex::Real>(advance_diag.material_substeps));
        extra_metrics.emplace_back(
            "fluid_max_reaction_number",
            static_cast<amrex::Real>(
                advance_diag.fluid_max_reaction_number));
        extra_metrics.emplace_back(
            "em_scattered_energy_J",
            static_cast<amrex::Real>(
                advance_diag.maxwell_scattered_energy_j));
        extra_metrics.emplace_back(
            "em_energy_change_from_ambient_J",
            static_cast<amrex::Real>(
                advance_diag.maxwell_energy_change_from_ambient_j));
        extra_metrics.emplace_back(
            "em_joule_work_J",
            static_cast<amrex::Real>(advance_diag.maxwell_joule_work_j));
        extra_metrics.emplace_back(
            "em_poynting_out_J",
            static_cast<amrex::Real>(advance_diag.maxwell_poynting_out_j));
        extra_metrics.emplace_back(
            "em_energy_balance_residual_J",
            static_cast<amrex::Real>(
                advance_diag.maxwell_energy_balance_residual_j));
        extra_metrics.emplace_back(
            "maxwell_jz_moment_A_m",
            static_cast<amrex::Real>(advance_diag.maxwell_jz_moment_a_m));
    }
    std::string const table_class = m_interaction_tables.TableClass();
    double fluid_unlimited_outgoing_cfl = 0.0;
    double fluid_min_outflow_scale = 1.0;
    double fluid_limited_outflow_fraction = 0.0;
    double fluid_limited_electron_number_fraction = 0.0;
    if (m_advance) {
        auto const& drift = m_advance->Fluid(0).LastCarrierDriftDiagnostics();
        fluid_unlimited_outgoing_cfl = drift.unlimited_max_outgoing_cfl;
        fluid_min_outflow_scale =
            std::min(fluid_min_outflow_scale, drift.min_outflow_scale);
        fluid_limited_outflow_fraction = drift.limited_outflow_fraction;
        fluid_limited_electron_number_fraction =
            drift.limited_electron_number_fraction;
    }

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }
    std::string const path = join_path(dir, "rrea_reduced.csv");
    std::string const fixed_header =
        "step,time_s,injected_macro_count,native_seed_particle_count,"
        "injected_physical_weight,"
        "next_seed_event,injection_hook_count,hook_call_count,"
        "field_advance_count,fluid_unlimited_outgoing_cfl,"
        "fluid_min_outflow_scale,fluid_limited_outflow_fraction,"
        "fluid_limited_electron_number_fraction,"
        "kinetic_secondary_electron_count,"
        "kinetic_secondary_weight,photon_created_count,photon_created_weight,"
        "photon_interaction_count,photon_interaction_weight,"
        "positron_created_count,positron_created_weight,"
        "compton_interaction_count,compton_interaction_weight,"
        "photoelectric_interaction_count,photoelectric_interaction_weight,"
        "pair_production_count,pair_production_weight,"
        "pair_nuclear_count,pair_nuclear_weight,"
        "pair_triplet_count,pair_triplet_weight,"
        "pair_recoil_electron_energy_eV,pair_local_deposit_energy_eV,"
        "photon_seeded_electron_weight,photon_seeded_positron_weight,"
        "photon_absorbed_energy_eV,positron_transport_energy_loss_eV,"
        "positron_thermalized_count,"
        "transport_ionization_weight,transport_low_secondary_weight,"
        "transport_kinetic_secondary_weight,transport_energy_loss_eV,"
        "reduced_schema_version,rng_seed,transport_table_class,"
        "total_ne_low,total_n_pos,total_n_neg,"
        "N_low_e_real,N_pos_real,N_neg_real,N_kinetic_e_real,"
        "N_kinetic_e_ge_1MeV_real,N_photon_real,N_positron_real,"
        "N_inst_e_real,N_cumulative_e_created,rho_fluid_C,max_ne_low,"
        "field_fraction_mean_accel_region,"
        "field_fraction_min_accel_region,field_fraction_p10_accel_region,"
        "field_fraction_max_accel_region,"
        "fraction_acceleration_volume_below_half_field,"
        "acceleration_volume_m3,"
        "total_rho_fluid,max_sigma,min_tau_m,total_S_low_electron,"
        "total_S_positive_ion,total_S_direct_negative_ion,"
        "ionization_event_count,total_energy_loss_ev_m3";
    auto const fixed_columns = split_csv_line(fixed_header);
    std::set<std::string> reduced_column_names(
        fixed_columns.begin(), fixed_columns.end());
    for (auto const& metric : extra_metrics) {
        if (metric.first.empty()
            || !reduced_column_names.insert(metric.first).second) {
            amrex::Abort(
                "RREA reduced-diagnostic header contains an empty or duplicate column: "
                + metric.first);
        }
    }
    std::string full_header = fixed_header;
    for (auto const& metric : extra_metrics) {
        full_header += "," + metric.first;
    }
    bool const write_header = ArchiveCsvIfHeaderChanged(path, full_header, step);
    std::ofstream output(path, std::ios::app);
    if (!output) {
        amrex::FileOpenFailed(path);
    }
    if (write_header) {
        output << full_header << "\n";
    }
    output << std::setprecision(17)
           << step << ","
           << time_s << ","
           << m_injected_macro_count << ","
           << m_native_seed_particle_count << ","
           << m_injected_physical_weight << ","
           << m_next_seed_event << ","
           << m_injection_hook_count << ","
           << m_before_field_solve_count << ","
           << m_field_advance_count << ","
           << fluid_unlimited_outgoing_cfl << ","
           << fluid_min_outflow_scale << ","
           << fluid_limited_outflow_fraction << ","
           << fluid_limited_electron_number_fraction << ","
           << global_kinetic_secondary_electron_count << ","
           << global_kinetic_secondary_weight << ","
           << global_photon_created_count << ","
           << global_photon_created_weight << ","
           << global_photon_interaction_count << ","
           << global_photon_interaction_weight << ","
           << global_positron_created_count << ","
           << global_positron_created_weight << ","
           << global_compton_interaction_count << ","
           << global_compton_interaction_weight << ","
           << global_photoelectric_interaction_count << ","
           << global_photoelectric_interaction_weight << ","
           << global_pair_production_count << ","
           << global_pair_production_weight << ","
           << global_pair_nuclear_count << ","
           << global_pair_nuclear_weight << ","
           << global_pair_triplet_count << ","
           << global_pair_triplet_weight << ","
           << global_pair_recoil_electron_energy_eV << ","
           << global_pair_local_deposit_energy_eV << ","
           << global_photon_seeded_electron_weight << ","
           << global_photon_seeded_positron_weight << ","
           << global_photon_absorbed_energy_eV << ","
           << global_positron_transport_energy_loss_eV << ","
           << global_positron_thermalized_count << ","
           << global_transport_ionization_weight << ","
           << global_transport_low_secondary_weight << ","
           << global_transport_kinetic_secondary_weight << ","
           << global_transport_energy_loss_eV << ","
           << checkpoint_reduced_diagnostic_schema_version << ","
           << m_rng_seed << ","
           << table_class << ","
           << total_ne_low << ","
           << total_n_pos << ","
           << total_n_neg << ","
           << n_low_e_real << ","
           << n_pos_real << ","
           << n_neg_real << ","
           << n_kinetic_e_real << ","
           << n_kinetic_e_ge_1mev_real << ","
           << n_photon_real << ","
           << n_positron_real << ","
           << n_inst_e_real << ","
           << n_cumulative_e_created << ","
           << rho_fluid_c << ","
           << max_ne_low << ","
           << field_metrics.mean_fraction << ","
           << field_metrics.min_fraction << ","
           << field_metrics.p10_fraction << ","
           << field_metrics.max_fraction << ","
           << field_metrics.fraction_below_half << ","
           << field_metrics.volume_m3 << ","
           << total_rho_fluid << ","
           << max_sigma << ","
           << min_tau_m << ","
           << total_s_low_electron << ","
           << total_s_positive_ion << ","
           << total_s_direct_negative_ion << ","
           << ionization_event_count << ","
           << total_energy_loss_ev_m3;
    for (auto const& metric : extra_metrics) {
        output << "," << metric.second;
    }
    output << "\n";

    // dM/dz side-car next to the reduced CSV
    // (NOT reduced-CSV columns: 3x128 bins do not belong in the flat row and
    // its fail-closed header).  Same restart semantics: append + exact
    // header match, re-verified on every append so a file replaced mid-run
    // is caught too.
    if (!current_profile_fluid.empty() && !m_video_diag_output_dir.empty()) {
        std::string const profile_path =
            join_path(m_video_diag_output_dir, "current_profile.csv");
        std::string profile_header = "step,time_s,z_lo_m,z_hi_m,nbins";
        for (char const* prefix :
             {"jz_fluid_A_m_b", "jz_kinetic_e_A_m_b", "jz_positron_A_m_b"}) {
            for (int b = 0; b < kRreaCurrentProfileBins; ++b) {
                profile_header += ",";
                profile_header += prefix;
                profile_header += std::to_string(b);
            }
        }
        // IOProcessor-only region (guard above): the collective mkdir would
        // deadlock here.  Rank-local creation is safe and idempotent.
        if (!amrex::UtilCreateDirectory(m_video_diag_output_dir, 0755)) {
            amrex::CreateDirectoryFailed(m_video_diag_output_dir);
        }
        std::ofstream profile(profile_path, std::ios::app);
        if (!profile) {
            amrex::FileOpenFailed(profile_path);
        }
        if (ArchiveCsvIfHeaderChanged(profile_path, profile_header, step)) {
            profile << profile_header << "\n";
        }
        profile << std::setprecision(17)
                << step << "," << time_s << ","
                << current_profile_z_lo << "," << current_profile_z_hi << ","
                << kRreaCurrentProfileBins;
        for (auto const* vec : {&current_profile_fluid,
                                &current_profile_kinetic_e,
                                &current_profile_positron}) {
            for (auto const value : *vec) {
                profile << "," << value;
            }
        }
        profile << "\n";
    }
}

void RreaWarpXCoupling::WriteDiagnostics(
    std::string const& dir,
    int step,
    amrex::Real time_s)
{
    if (!m_initialized) {
        return;
    }
    ++m_diagnostics_write_count;
    // Mesh trees are inode-heavy (one file per box per field), so
    // rrea.mesh_diag_interval decouples them
    // from the reduced-CSV cadence.  -1 (default): every diag write, unchanged.
    // 0: never.  N > 0: only when step % N == 0.
    bool const write_mesh_tree =
        m_mesh_diag_interval < 0
        || (m_mesh_diag_interval > 0 && step % m_mesh_diag_interval == 0);
    if (write_mesh_tree) {
        std::string const diag_root = join_path(dir, "rrea_diag");
        collective_create_dir_all(diag_root);
        std::string const step_root = join_path(diag_root, step_dir_name(step));
        WriteMeshFields(step_root);
        WriteMetadata(step_root);
    }
    AppendReducedDiagnostics(dir, step, time_s);
}

}  // namespace rrea::warpx
