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
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rrea::warpx {

using detail::collective_create_dir_all;
using detail::global_count_sum;
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

amrex::Real max_field(amrex::MultiFab const& mf)
{
    return mf.max(0, 0, false);
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
        auto const arr = mf.const_array(mfi);
        amrex::LoopOnCpu(mfi.tilebox(), [&](int i, int j, int k) noexcept {
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

// Signed vertical current moment of one kinetic species above an energy,
// M_z = q * sum_i w_i v_z,i [A*m] over K_i >= threshold_eV, with v_z = u_z /
// gamma (u is the stored proper velocity).  Recorded for the offline radio
// waveform reconstruction (architecture guide, "From Simulation to Radio, Longitudinal-Field and
// Optical Observables");
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
            if (p.KineticEnergyEv() < threshold_eV) {
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
    auto const& fluid = m_advance->Fluid();
    auto const& geom0 = m_level.geom;
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
    amrex::Real const max_ne_low = max_field(fluid.LowElectronDensity());
    amrex::Real const max_sigma = max_field(fluid.Conductivity());
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
        *m_level.efield_z,
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
        *m_level.efield_x,
        *m_level.efield_z,
        m_level.density_ratio,
        amrex::Real(m_advance_config.fluid.transport_density_ratio),
        fluid.LowElectronDensity(),
        m_events->LowElectronSourceRate(),
        geom0,
        profile_band_valid ? m_profile_active_z_min_m : amrex::Real(0.0),
        profile_band_valid ? m_profile_active_z_max_m
                           : amrex::Real(field_init.acceleration_z_max_m),
        field_init.TaperActive()
            ? amrex::Real(field_init.taper_r_start_m)
            : std::numeric_limits<amrex::Real>::infinity());
    double fluid_unlimited_outgoing_cfl = 0.0;
    double fluid_min_outflow_scale = 1.0;
    double fluid_limited_outflow_fraction = 0.0;
    double fluid_limited_electron_number_fraction = 0.0;
    if (m_advance) {
        auto const& drift = m_advance->Fluid().LastCarrierDriftDiagnostics();
        fluid_unlimited_outgoing_cfl = drift.unlimited_max_outgoing_cfl;
        fluid_min_outflow_scale =
            std::min(fluid_min_outflow_scale, drift.min_outflow_scale);
        fluid_limited_outflow_fraction = drift.limited_outflow_fraction;
        fluid_limited_electron_number_fraction =
            drift.limited_electron_number_fraction;
    }
    // One ordered column list owns the header and the row.  Rank-local
    // counters are marked and summed by one batched allreduce below.
    struct Column {
        Column(std::string n, amrex::Real v, bool local = false)
            : name(std::move(n)), value(v), rank_local(local) {}
        std::string name;
        amrex::Real value;
        bool rank_local;
    };
    auto const local = [](auto value) { return static_cast<amrex::Real>(value); };
    std::vector<Column> columns = {
        {"step", local(step)},
        {"time_s", time_s},
        {"injected_macro_count", local(m_injected_macro_count)},
        {"native_seed_particle_count", local(m_native_seed_particle_count)},
        {"injected_physical_weight", m_injected_physical_weight},
        {"next_seed_event", local(m_next_seed_event)},
        {"fluid_unlimited_outgoing_cfl", fluid_unlimited_outgoing_cfl},
        {"fluid_min_outflow_scale", fluid_min_outflow_scale},
        {"fluid_limited_outflow_fraction", fluid_limited_outflow_fraction},
        {"fluid_limited_electron_number_fraction",
         fluid_limited_electron_number_fraction},
        {"kinetic_secondary_electron_count",
         local(m_kinetic_secondary_electron_count), true},
        {"kinetic_secondary_weight", m_kinetic_secondary_weight, true},
        {"photon_created_count", local(m_photon_created_count), true},
        {"photon_created_weight", m_photon_created_weight, true},
        {"photon_interaction_count", local(m_ledger.photon_interaction_count), true},
        {"photon_interaction_weight", m_ledger.photon_interaction_weight, true},
        {"positron_created_count", local(m_positron_created_count), true},
        {"positron_created_weight", m_positron_created_weight, true},
        {"compton_interaction_count", local(m_ledger.compton_interaction_count), true},
        {"compton_interaction_weight", m_ledger.compton_interaction_weight, true},
        {"photoelectric_interaction_count",
         local(m_ledger.photoelectric_interaction_count), true},
        {"photoelectric_interaction_weight",
         m_ledger.photoelectric_interaction_weight, true},
        {"pair_production_count", local(m_ledger.pair_production_count), true},
        {"pair_production_weight", m_ledger.pair_production_weight, true},
        {"pair_nuclear_count", local(m_ledger.pair_nuclear_count), true},
        {"pair_nuclear_weight", m_ledger.pair_nuclear_weight, true},
        {"pair_triplet_count", local(m_ledger.pair_triplet_count), true},
        {"pair_triplet_weight", m_ledger.pair_triplet_weight, true},
        {"pair_recoil_electron_energy_eV", m_ledger.pair_recoil_electron_energy_eV, true},
        {"pair_local_deposit_energy_eV", m_ledger.pair_local_deposit_energy_eV, true},
        {"photon_seeded_electron_weight", m_ledger.photon_seeded_electron_weight, true},
        {"photon_seeded_positron_weight", m_ledger.photon_seeded_positron_weight, true},
        {"photon_absorbed_energy_eV", m_ledger.photon_absorbed_energy_eV, true},
        {"positron_transport_energy_loss_eV",
         m_ledger.positron_transport_energy_loss_eV, true},
        {"positron_thermalized_count", local(m_ledger.positron_thermalized_count), true},
        {"transport_ionization_weight", m_ledger.transport_ionization_weight, true},
        {"transport_low_secondary_weight", m_ledger.transport_low_secondary_weight, true},
        {"transport_kinetic_secondary_weight",
         m_ledger.transport_kinetic_secondary_weight, true},
        {"transport_energy_loss_eV", m_ledger.transport_energy_loss_eV, true},
        {"N_low_e_real", n_low_e_real},
        {"N_pos_real", n_pos_real},
        {"N_neg_real", n_neg_real},
        {"N_kinetic_e_real", n_kinetic_e_real},
        {"N_kinetic_e_ge_1MeV_real", n_kinetic_e_ge_1mev_real},
        {"N_photon_real", n_photon_real},
        {"N_positron_real", n_positron_real},
        {"rho_fluid_C", rho_fluid_c},
        {"max_ne_low", max_ne_low},
        {"field_fraction_mean_accel_region", field_metrics.mean_fraction},
        {"field_fraction_min_accel_region", field_metrics.min_fraction},
        {"field_fraction_p10_accel_region", field_metrics.p10_fraction},
        {"field_fraction_max_accel_region", field_metrics.max_fraction},
        {"fraction_acceleration_volume_below_half_field",
         field_metrics.fraction_below_half},
        {"acceleration_volume_m3", field_metrics.volume_m3},
        {"max_sigma", max_sigma}};
    columns.reserve(columns.size() + 128);
    // Global vertical current moments [A*m] for offline radio-waveform
    // reconstruction.  The fluid moment is the step-averaged fluid face
    // current Maxwell received (drift and diffusion of every carrier), the
    // current that closes charge continuity with the field solve; the
    // sigma|Ez|-weighted rms extents only size its region.  The kinetic
    // moments are exact per-macro sums q*w*v_z over every energy, so the
    // >=1 MeV column directly validates the offline -e*N*<beta_z>*c
    // approximation.  Diagnostics only; nothing feeds back into the physics.
    amrex::Real const current_profile_z_lo = geom0.ProbLo(1);
    amrex::Real const current_profile_z_hi = geom0.ProbHi(1);
    std::vector<amrex::Real> current_profile_fluid =
        m_advance->FluidCurrentMomentProfile(kRreaCurrentProfileBins);
    std::vector<amrex::Real> current_profile_kinetic_e(kRreaCurrentProfileBins, amrex::Real(0.0));
    std::vector<amrex::Real> current_profile_positron(kRreaCurrentProfileBins, amrex::Real(0.0));
    {
        static constexpr amrex::Real elementary_charge_C = amrex::Real(rrea::qe);
        auto const total = [](std::vector<amrex::Real> const& bins) {
            return std::accumulate(bins.begin(), bins.end(), amrex::Real(0.0));
        };
        auto const global_ohmic = ohmic_discharge_metrics_rz(
            *m_level.efield_z,
            fluid.Conductivity(),
            geom0,
            geom0.ProbLo(1),
            geom0.ProbHi(1),
            std::numeric_limits<amrex::Real>::max());
        columns.emplace_back("global_jz_fluid_A_m", total(current_profile_fluid));
        columns.emplace_back(
            "fluid_current_rms_r_m", global_ohmic.fluid_current_rms_r_m);
        columns.emplace_back(
            "fluid_current_rms_z_m", global_ohmic.fluid_current_rms_z_m);
        // Rank-local z-binned kinetic moments, one batched allreduce; each
        // global kinetic moment is the sum of its bins.
        amrex::Real const inv_dz_bin =
            static_cast<amrex::Real>(kRreaCurrentProfileBins)
            / (current_profile_z_hi - current_profile_z_lo);
        species_current_profile_z(
            m_seed_species_name, -elementary_charge_C,
            current_profile_z_lo, inv_dz_bin, current_profile_kinetic_e);
        species_current_profile_z(
            m_positron_species_name, elementary_charge_C,
            current_profile_z_lo, inv_dz_bin, current_profile_positron);
        std::vector<amrex::Real> kinetic_bins = current_profile_kinetic_e;
        kinetic_bins.insert(kinetic_bins.end(),
                            current_profile_positron.begin(),
                            current_profile_positron.end());
        global_vector_sum(kinetic_bins);
        std::copy(kinetic_bins.begin(),
                  kinetic_bins.begin() + kRreaCurrentProfileBins,
                  current_profile_kinetic_e.begin());
        std::copy(kinetic_bins.begin() + kRreaCurrentProfileBins,
                  kinetic_bins.end(),
                  current_profile_positron.begin());
        columns.emplace_back(
            "global_jz_kinetic_e_A_m", total(current_profile_kinetic_e));
        columns.emplace_back(
            "global_jz_kinetic_e_ge_1MeV_A_m",
            species_current_moment_z_A_m(
                m_seed_species_name, -elementary_charge_C, amrex::Real(1.0e6)));
        columns.emplace_back(
            "global_jz_positron_A_m", total(current_profile_positron));
    }
    amrex::Real const photon_alive_energy_eV = photon_species_energy(m_photon_species_name);
    std::vector<amrex::Real> photon_ledger_flows = {
        m_ledger.photon_escaped_energy_eV,
        m_ledger.photon_to_charged_energy_eV,
        m_ledger.photon_local_deposit_energy_eV,
        m_ledger.total_brems_photon_energy_eV,
        m_positron_annihilation_photon_energy_eV};
    global_vector_sum(photon_ledger_flows);
    amrex::Real const global_total_brems_photon_energy_eV = photon_ledger_flows[3];
    amrex::Real const global_positron_annihilation_photon_energy_eV =
        photon_ledger_flows[4];
    RreaPhotonLedgerFlows const photon_ledger{
        m_initial_photon_alive_energy_eV,
        global_total_brems_photon_energy_eV,
        global_positron_annihilation_photon_energy_eV,
        photon_alive_energy_eV,
        photon_ledger_flows[0],
        photon_ledger_flows[1],
        photon_ledger_flows[2]};
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
                << ", brems_created_eV=" << global_total_brems_photon_energy_eV
                << ", annihilation_created_eV="
                << global_positron_annihilation_photon_energy_eV
                << ", alive_eV=" << photon_alive_energy_eV
                << ", escaped_eV=" << photon_ledger_flows[0]
                << ", to_charged_eV=" << photon_ledger_flows[1]
                << ", local_deposit_eV=" << photon_ledger_flows[2];
        amrex::Abort(message.str());
    }
    columns.emplace_back(
        "photon_continuous_soft_brems_energy_eV",
        m_ledger.photon_continuous_soft_brems_energy_eV, true);
    // Realized reduced-field columns (see en_td_region_metrics_rz): three
    // weightings plus the de Urquijo certified-range occupation fractions.
    columns.emplace_back("en_td_accel_min", en_metrics.min_en_td);
    columns.emplace_back("en_td_accel_max", en_metrics.max_en_td);
    {
        static constexpr std::array<char const*, 3> channel_names = {
            "vol", "ne_low", "src"};
        static constexpr std::array<char const*, 3> quantile_names = {
            "p10", "p50", "p90"};
        for (std::size_t channel = 0; channel < channel_names.size();
             ++channel) {
            columns.emplace_back(
                std::string("en_td_accel_") + channel_names[channel]
                    + "_weighted_mean",
                en_metrics.weighted_mean_en_td[channel]);
            for (std::size_t q = 0; q < quantile_names.size(); ++q) {
                columns.emplace_back(
                    std::string("en_td_accel_") + channel_names[channel]
                        + "_" + quantile_names[q],
                    en_metrics.p10_p50_p90[channel][q]);
            }
        }
    }
    columns.emplace_back(
        "en_td_accel_vol_frac_below_3td", en_metrics.vol_frac_below_3td);
    columns.emplace_back(
        "en_td_accel_vol_frac_above_30td", en_metrics.vol_frac_above_30td);
    columns.emplace_back(
        "en_td_accel_ne_low_frac_below_3td", en_metrics.ne_frac_below_3td);
    if (m_advance_config.electron_closure != nullptr
        && m_advance_config.electron_closure->Loaded()) {
        // Whole-domain realized range and held-cell count from the en_table
        // evaluations the step actually consumed (see
        // RreaAmrexAdvanceDiagnostics); above-range/invalid states abort, so
        // their absence here is load-bearing, not optimistic.
        auto const& advance_diag = m_advance->LastDiagnostics();
        columns.emplace_back(
            "closure_realized_min_en_td",
            advance_diag.closure_evaluated
                ? static_cast<amrex::Real>(advance_diag.closure_min_en_td)
                : std::numeric_limits<amrex::Real>::quiet_NaN());
        columns.emplace_back(
            "closure_realized_max_en_td",
            advance_diag.closure_evaluated
                ? static_cast<amrex::Real>(advance_diag.closure_max_en_td)
                : std::numeric_limits<amrex::Real>::quiet_NaN());
        columns.emplace_back(
            "closure_held_below_cells",
            static_cast<amrex::Real>(
                advance_diag.closure_held_below_cells));
        columns.emplace_back(
            "closure_held_below_electron_fraction",
            static_cast<amrex::Real>(
                advance_diag.closure_max_held_below_electron_fraction));
        columns.emplace_back(
            "closure_held_below_source_fraction",
            static_cast<amrex::Real>(
                advance_diag.closure_max_held_below_source_fraction));
        columns.emplace_back(
            "closure_held_below_conductivity_fraction",
            static_cast<amrex::Real>(
                advance_diag.closure_max_held_below_conductivity_fraction));
    }

    if (m_population_ceiling_policy == "adaptive_resample_v1") {
        // adaptive_resample_v1 population-control ledger (global cumulative).
        // "killed" records every negative weight increment (including partial
        // reductions), while "boost" records every positive increment.  The
        // CIC null-space move balances their total to deposition round-off in
        // each realization.
        columns.insert(columns.end(), {
            {"resample_killed_weight", m_resample_killed_weight, true},
            {"resample_boost_weight", m_resample_boost_weight, true},
            {"resample_split_weight", m_resample_split_weight, true},
            {"resample_killed_energy_eV", m_resample_killed_energy_eV, true},
            {"resample_boost_energy_eV", m_resample_boost_energy_eV, true},
            {"resample_killed_count", local(m_resample_killed_count), true},
            {"resample_split_count", local(m_resample_split_count), true}});
        columns.emplace_back(
            "resample_thin_rounds",
            static_cast<amrex::Real>(m_resample_thin_rounds));
        columns.emplace_back(
            "resample_electron_exact_capacity",
            static_cast<amrex::Real>(m_resample_exact_capacity[0]));
        columns.emplace_back(
            "resample_electron_geometric_floor",
            static_cast<amrex::Real>(m_resample_geometric_floor[0]));
        columns.emplace_back(
            "resample_positron_exact_capacity",
            static_cast<amrex::Real>(m_resample_exact_capacity[1]));
        columns.emplace_back(
            "resample_positron_geometric_floor",
            static_cast<amrex::Real>(m_resample_geometric_floor[1]));
    }
    static constexpr std::array<char const*, 3> particle_labels = {
        "kinetic_e", "photon", "positron"};
    for (std::size_t i = 0; i < particle_weight_stats.size(); ++i) {
        columns.emplace_back(
            std::string(particle_labels[i]) + "_N_eff",
            particle_weight_stats[i].EffectiveSampleSize());
        columns.emplace_back(
            std::string(particle_labels[i]) + "_max_weight",
            particle_weight_stats[i].max_weight);
    }
    columns.insert(columns.end(), {
        {"ion_pair_source_weight_cumulative", m_ion_pair_source_weight, true},
        {"ion_pair_source_z_weighted_m_cumulative", m_ion_pair_source_z_weighted_m, true},
        {"hard_moller_event_count", local(m_ledger.hard_moller_event_count), true},
        {"electron_brems_photon_count", local(m_ledger.electron_brems_photon_count), true},
        {"N_kinetic_e_ge_100keV_real", n_kinetic_e_ge_100kev_real},
        {"N_positron_ge_100keV_real", n_positron_ge_100kev_real},
        {"photon_seeded_kinetic_electron_weight",
         m_ledger.photon_seeded_kinetic_electron_weight, true},
        {"photon_seeded_low_electron_weight",
         m_ledger.photon_seeded_low_electron_weight, true},
        {"photon_seeded_kinetic_positron_weight",
         m_ledger.photon_seeded_kinetic_positron_weight, true},
        {"photon_seeded_below_cutoff_positron_weight",
         m_ledger.photon_seeded_below_cutoff_positron_weight, true},
        {"positron_bhabha_event_count", local(m_ledger.positron_bhabha_event_count), true},
        {"positron_brems_event_count", local(m_ledger.positron_brems_event_count), true},
        {"positron_annihilation_count", local(m_positron_annihilation_count), true},
        {"positron_annihilation_at_rest_count",
         local(m_positron_annihilation_at_rest_count), true},
        {"positron_annihilation_in_flight_count",
         local(m_positron_annihilation_in_flight_count), true}});
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
        columns.emplace_back("netchord_segment_count", netchord_sums[0]);
        columns.emplace_back(
            "netchord_jz_moment_A_m",
            m_netchord_dt_s > amrex::Real(0.0)
                ? qe_C * netchord_sums[1] / m_netchord_dt_s
                : amrex::Real(0.0));
        columns.emplace_back(
            "netchord_created_charge_e", netchord_sums[2]);
        columns.emplace_back(
            "netchord_escaped_charge_e", netchord_sums[3]);
        columns.emplace_back(
            "netchord_demoted_charge_e", netchord_sums[4]);
        columns.emplace_back(
            "netchord_annihilated_charge_e", netchord_sums[5]);
    }
    // Maxwell TM columns: the eps0-Gauss monitor (current-free seed-injection
    // baseline), the peak Btheta available to near-to-far analysis, the
    // scattered field energy, and the physical total-field energy change
    // relative to the ambient (which includes the ambient/scattered cross
    // term).
    {
        auto const& advance_diag = m_advance->LastDiagnostics();
        columns.emplace_back(
            "em_gauss_residual_rel",
            static_cast<amrex::Real>(
                advance_diag.maxwell_gauss_residual_normalized));
        columns.emplace_back(
            "em_charge_continuity_max_rel",
            static_cast<amrex::Real>(
                advance_diag.maxwell_charge_continuity_max_relative));
        columns.emplace_back(
            "em_charge_continuity_max_abs_e_per_cell",
            static_cast<amrex::Real>(
                advance_diag.maxwell_charge_continuity_max_abs_e_per_cell));
        columns.emplace_back(
            "em_charge_continuity_signed_accum_e",
            static_cast<amrex::Real>(
                advance_diag.maxwell_charge_continuity_signed_accum_e));
        columns.emplace_back(
            "em_btheta_max_T",
            static_cast<amrex::Real>(advance_diag.maxwell_btheta_max_t));
        columns.emplace_back(
            "em_max_dt_sigma_face_over_eps0",
            static_cast<amrex::Real>(
                advance_diag.maxwell_max_dt_sigma_face_over_eps0));
        columns.emplace_back(
            "fluid_explicit_diffusion_number",
            static_cast<amrex::Real>(
                advance_diag.fluid_max_explicit_diffusion_number));
        columns.emplace_back(
            "material_substeps",
            static_cast<amrex::Real>(advance_diag.material_substeps));
        columns.emplace_back(
            "fluid_max_reaction_number",
            static_cast<amrex::Real>(
                advance_diag.fluid_max_reaction_number));
        columns.emplace_back(
            "em_scattered_energy_J",
            static_cast<amrex::Real>(
                advance_diag.maxwell_scattered_energy_j));
        columns.emplace_back(
            "em_energy_change_from_ambient_J",
            static_cast<amrex::Real>(
                advance_diag.maxwell_energy_change_from_ambient_j));
        columns.emplace_back(
            "em_joule_work_J",
            static_cast<amrex::Real>(advance_diag.maxwell_joule_work_j));
        columns.emplace_back(
            "em_poynting_out_J",
            static_cast<amrex::Real>(advance_diag.maxwell_poynting_out_j));
        columns.emplace_back(
            "em_energy_balance_residual_J",
            static_cast<amrex::Real>(
                advance_diag.maxwell_energy_balance_residual_j));
        columns.emplace_back(
            "maxwell_jz_moment_A_m",
            static_cast<amrex::Real>(advance_diag.maxwell_jz_moment_a_m));
    }
    {
        std::vector<amrex::Real> sums;
        for (auto const& column : columns) {
            if (column.rank_local) {
                sums.push_back(column.value);
            }
        }
        global_vector_sum(sums);
        std::size_t next = 0;
        for (auto& column : columns) {
            if (column.rank_local) {
                column.value = sums[next++];
            }
        }
    }

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }
    std::string const path = join_path(dir, "rrea_reduced.csv");
    std::set<std::string> reduced_column_names;
    std::string full_header;
    for (auto const& metric : columns) {
        if (metric.name.empty()
            || !reduced_column_names.insert(metric.name).second) {
            amrex::Abort(
                "RREA reduced-diagnostic header contains an empty or duplicate column: "
                + metric.name);
        }
        full_header += (full_header.empty() ? "" : ",") + metric.name;
    }
    bool const write_header = ArchiveCsvIfHeaderChanged(path, full_header, step);
    std::ofstream output(path, std::ios::app);
    if (!output) {
        amrex::FileOpenFailed(path);
    }
    if (write_header) {
        output << full_header << "\n";
    }
    output << std::setprecision(17);
    for (std::size_t c = 0; c < columns.size(); ++c) {
        output << (c == 0 ? "" : ",") << columns[c].value;
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
    // Mesh trees are inode-heavy (one file per box per field), so
    // rrea.mesh_diag_interval decouples them
    // from the reduced-CSV cadence.  -1: every diag write; 0 (default): never;
    // N > 0: only when step % N == 0.
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
