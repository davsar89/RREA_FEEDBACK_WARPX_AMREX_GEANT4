// Particle and event sources for RreaWarpXCoupling: scheduled seed injection,
// the one AddNParticles secondary batch commit behind the three Create* entry
// points, and the fluid/ion event-source deposits.

#include "RreaWarpXCoupling.H"

#include "RreaCouplingDetail.H"
#include "RreaParticleInteraction.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace rrea::warpx {

namespace detail {


int append_rrea_previous_position_attributes(
    WarpXParticleContainer& species,
    amrex::Vector<amrex::ParticleReal> const& x,
    amrex::Vector<amrex::ParticleReal> const& y,
    amrex::Vector<amrex::ParticleReal> const& z,
    amrex::Vector<amrex::ParticleReal> const& ux,
    amrex::Vector<amrex::ParticleReal> const& uy,
    amrex::Vector<amrex::ParticleReal> const& uz,
    amrex::Vector<amrex::Vector<amrex::ParticleReal>>& attr_real)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(species, x, y, z, ux, uy, uz, attr_real);
    amrex::Abort("RREA previous-position spawn initialization requires WarpX RZ");
    return 0;
#else
    if (!species.HasRealComp("rrea_prev_r")
        || !species.HasRealComp("rrea_prev_z")
        || !species.HasRealComp("rrea_prev_theta")
        || !species.HasRealComp("rrea_prev_ux")
        || !species.HasRealComp("rrea_prev_uy")
        || !species.HasRealComp("rrea_prev_uz")) {
        amrex::Abort(
            "RREA particle creation requires previous position/momentum runtime components");
    }
    int const first = PIdx::nattribs;
    if (species.GetRealCompIndex("rrea_prev_r") != first
        || species.GetRealCompIndex("rrea_prev_z") != first + 1
        || species.GetRealCompIndex("rrea_prev_theta") != first + 2
        || species.GetRealCompIndex("rrea_prev_ux") != first + 3
        || species.GetRealCompIndex("rrea_prev_uy") != first + 4
        || species.GetRealCompIndex("rrea_prev_uz") != first + 5) {
        amrex::Abort(
            "RREA previous position/momentum runtime components must be the first six "
            "runtime real attributes so AddNParticles can initialize them atomically");
    }
    if (x.size() != y.size() || x.size() != z.size() || x.size() != ux.size()
        || x.size() != uy.size() || x.size() != uz.size()) {
        amrex::Abort("RREA previous-state spawn arrays have inconsistent sizes");
    }
    amrex::Vector<amrex::ParticleReal> previous_r(x.size());
    amrex::Vector<amrex::ParticleReal> previous_z(z);
    amrex::Vector<amrex::ParticleReal> previous_theta(x.size());
    for (amrex::Long i = 0; i < x.size(); ++i) {
        previous_r[i] = std::hypot(x[i], y[i]);
        previous_theta[i] = std::atan2(y[i], x[i]);
    }
    attr_real.push_back(std::move(previous_r));
    attr_real.push_back(std::move(previous_z));
    attr_real.push_back(std::move(previous_theta));
    attr_real.push_back(ux);
    attr_real.push_back(uy);
    attr_real.push_back(uz);
    // DefaultInitializeRuntimeAttributes fills only QED optical depths and
    // ParmParse attributes. Zero-fill all remaining runtime components;
    // callers overwrite any column that needs a nonzero value.
    int const total_runtime_real = species.NumRealComps() - PIdx::nattribs;
    for (int extra = 6; extra < total_runtime_real; ++extra) {
        attr_real.push_back(amrex::Vector<amrex::ParticleReal>(
            x.size(), amrex::ParticleReal(0.0)));
    }
    // One built-in attribute (weight) plus every runtime attribute.
    return 1 + total_runtime_real;
#endif
}

}  // namespace detail

using detail::append_rrea_previous_position_attributes;

void RreaWarpXCoupling::InjectScheduledSeeds(
    WarpX& warpx,
    int step,
    amrex::Real time_s,
    amrex::Real dt_s)
{
    ReadParameters();
    if (!m_enabled) {
        return;
    }
    InitFromWarpX(warpx);
    ++m_injection_hook_count;
    if (m_debug_actual_continuity_fixture) {
        InjectActualContinuityFixtureParticles(warpx, step);
    }

    if (dt_s <= amrex::Real(0.0) || m_seed_schedule.Empty()) {
        amrex::ignore_unused(step);
        return;
    }

    // One buffer set per seed species: index 0 electrons, 1 positrons.
    struct SeedBuffers {
        amrex::Vector<amrex::ParticleReal> x, y, z, ux, uy, uz, weights;
    };
    std::array<SeedBuffers, 2> buffers;

    amrex::Real const event_window_end = time_s + dt_s;
    while (m_next_seed_event < m_seed_schedule.Size()) {
        auto const& event = m_seed_schedule.Event(m_next_seed_event);
        // Window and momentum rules are single primitives pinned by the
        // seed-schedule smoke -- do not reimplement them inline.
        auto const window_class = rrea::RreaSeedEventWindowClass(
            event.time_s, time_s, event_window_end);
        if (window_class == rrea::RreaSeedWindowClass::SkipStale) {
            ++m_next_seed_event;
            continue;
        }
        if (window_class == rrea::RreaSeedWindowClass::BeyondWindow) {
            break;
        }
        auto& buffer =
            buffers[event.species_id == rrea::RreaSeedSpeciesPositron ? 1 : 0];
        buffer.x.push_back(static_cast<amrex::ParticleReal>(event.x_m));
        buffer.y.push_back(static_cast<amrex::ParticleReal>(event.y_m));
        buffer.z.push_back(static_cast<amrex::ParticleReal>(event.z_m));
        // Same e+- rest mass, so one momentum primitive serves both species.
        auto const momentum = rrea::RreaNormalizeSeedMomentum(
            event.ux, event.uy, event.uz, event.kinetic_energy_eV);
        buffer.ux.push_back(static_cast<amrex::ParticleReal>(momentum.ux));
        buffer.uy.push_back(static_cast<amrex::ParticleReal>(momentum.uy));
        buffer.uz.push_back(static_cast<amrex::ParticleReal>(momentum.uz));
        buffer.weights.push_back(
            static_cast<amrex::ParticleReal>(event.weight_real_electrons));
        m_injected_physical_weight += event.weight_real_electrons;
        ++m_next_seed_event;
    }

    long const n_electron = static_cast<long>(buffers[0].x.size());
    long const n_positron = static_cast<long>(buffers[1].x.size());
    if (n_electron + n_positron == 0) {
        return;
    }

    struct SeedTarget {
        std::string const& species_name;
        long max_macros;
        SeedBuffers& buffer;
        long count;
    };
    std::array<SeedTarget, 2> const targets = {
        SeedTarget{m_seed_species_name, m_max_electron_macros, buffers[0],
                   n_electron},
        SeedTarget{m_positron_species_name, m_max_positron_macros, buffers[1],
                   n_positron},
    };
    for (auto const& target : targets) {
        if (target.count == 0) {
            continue;
        }
        auto& container = warpx.GetPartContainer()
            .GetParticleContainerFromName(target.species_name);
        EnforcePopulationCeiling(
            warpx,
            target.species_name,
            target.max_macros,
            static_cast<amrex::Long>(target.count),
            "scheduled_seed",
            step);
        amrex::Vector<amrex::Vector<amrex::ParticleReal>> attr_real;
        attr_real.push_back(target.buffer.weights);
        int const nattr_real = append_rrea_previous_position_attributes(
            container, target.buffer.x, target.buffer.y, target.buffer.z,
            target.buffer.ux, target.buffer.uy, target.buffer.uz, attr_real);
        amrex::Vector<amrex::Vector<int>> attr_int;
        container.AddNParticles(
            0,
            target.count,
            target.buffer.x,
            target.buffer.y,
            target.buffer.z,
            target.buffer.ux,
            target.buffer.uy,
            target.buffer.uz,
            nattr_real,
            attr_real,
            0,
            attr_int,
            // uniqueparticles=0 partitions the global seed list across ranks.
            0);
    }

    m_injected_macro_count +=
        static_cast<std::uint64_t>(n_electron + n_positron);
    m_maxwell_continuity_reprime_required = true;
    auto const native_count = warpx.GetPartContainer()
        .GetParticleContainerFromName(m_seed_species_name)
        .TotalNumberOfParticles();
    m_native_seed_particle_count = static_cast<std::uint64_t>(native_count);
    amrex::Print() << "RREA injected native seeds: step=" << step
                   << ", time_s=" << time_s
                   << ", electron_added=" << n_electron
                   << ", positron_added=" << n_positron
                   << ", injected_macro_count=" << m_injected_macro_count
                   << ", injected_physical_weight=" << m_injected_physical_weight
                   << ", native_" << m_seed_species_name << "_count=" << native_count
                   << "\n";
}

// ONE batch-commit primitive for every locally created macroparticle -- the
// electron/positron/photon Create* entry points below are thin species
// policies over it (species container, ceiling, momentum mapping, counters).
// This is also the single home of the AddNParticles wiring (the runtime
// attribute-layout aborts in append_rrea_previous_position_attributes are
// its guard).  Momentum conventions: charged secondaries carry the proper
// velocity u(K) along the given direction (zero direction falls back to
// +z; K <= 0 keeps the raw components); photons carry p = E_gamma * dir in
// the eV momentum convention (zero direction falls back to +z) and a
// non-positive E_gamma is skipped.  All ranks call collectively so
// AddNParticles/Redistribute cannot diverge when only one rank sampled a
// secondary.
void RreaWarpXCoupling::CommitLocalParticleBatch(
    WarpX& warpx,
    std::string const& species_name,
    amrex::Long max_macros,
    char const* ceiling_context,
    bool photon_momentum,
    std::uint64_t& created_count,
    amrex::Real& created_weight,
    std::vector<amrex::Real> const& x_m,
    std::vector<amrex::Real> const& y_m,
    std::vector<amrex::Real> const& z_m,
    std::vector<amrex::Real> const& ux,
    std::vector<amrex::Real> const& uy,
    std::vector<amrex::Real> const& uz,
    std::vector<amrex::Real> const& weight,
    std::vector<amrex::Real> const& energy_eV,
    int step,
    bool population_ceiling_preflighted)
{
    std::size_t const count = x_m.size();
    if (y_m.size() != count || z_m.size() != count || ux.size() != count
        || uy.size() != count || uz.size() != count || weight.size() != count
        || energy_eV.size() != count) {
        amrex::Abort("RREA local particle batch has inconsistent vector sizes");
    }

    amrex::Vector<amrex::ParticleReal> x;
    amrex::Vector<amrex::ParticleReal> y;
    amrex::Vector<amrex::ParticleReal> z;
    amrex::Vector<amrex::ParticleReal> px;
    amrex::Vector<amrex::ParticleReal> py;
    amrex::Vector<amrex::ParticleReal> pz;
    amrex::Vector<amrex::ParticleReal> weights;
    x.reserve(count);
    y.reserve(count);
    z.reserve(count);
    px.reserve(count);
    py.reserve(count);
    pz.reserve(count);
    weights.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        if (weight[i] <= amrex::Real(0.0)
            || (photon_momentum && energy_eV[i] <= amrex::Real(0.0))) {
            continue;
        }
        amrex::Real px_value = ux[i];
        amrex::Real py_value = uy[i];
        amrex::Real pz_value = uz[i];
        amrex::Real const direction_norm = std::sqrt(
            px_value * px_value + py_value * py_value + pz_value * pz_value);
        if (photon_momentum) {
            amrex::Real const scale = direction_norm > amrex::Real(0.0)
                ? energy_eV[i] / direction_norm
                : energy_eV[i];
            px_value = ux[i] * scale;
            py_value = uy[i] * scale;
            pz_value = direction_norm > amrex::Real(0.0)
                ? uz[i] * scale
                : energy_eV[i];
        } else if (energy_eV[i] > amrex::Real(0.0)) {
            amrex::Real const u_mag =
                rrea::RreaElectronProperVelocityFromKineticEv(energy_eV[i]);
            if (direction_norm > amrex::Real(0.0)) {
                px_value = u_mag * px_value / direction_norm;
                py_value = u_mag * py_value / direction_norm;
                pz_value = u_mag * pz_value / direction_norm;
            } else {
                px_value = amrex::Real(0.0);
                py_value = amrex::Real(0.0);
                pz_value = u_mag;
            }
        }
        x.push_back(static_cast<amrex::ParticleReal>(x_m[i]));
        y.push_back(static_cast<amrex::ParticleReal>(y_m[i]));
        z.push_back(static_cast<amrex::ParticleReal>(z_m[i]));
        px.push_back(static_cast<amrex::ParticleReal>(px_value));
        py.push_back(static_cast<amrex::ParticleReal>(py_value));
        pz.push_back(static_cast<amrex::ParticleReal>(pz_value));
        weights.push_back(static_cast<amrex::ParticleReal>(weight[i]));
    }

    amrex::Long global_spawn_count = static_cast<amrex::Long>(x.size());
    amrex::ParallelDescriptor::ReduceLongSum(global_spawn_count);
    if (global_spawn_count == 0) {
        return;
    }

    auto& species =
        warpx.GetPartContainer().GetParticleContainerFromName(species_name);

    if (!population_ceiling_preflighted) {
        EnforcePopulationCeiling(
            warpx,
            species_name,
            max_macros,
            global_spawn_count,
            ceiling_context,
            step);
    }

    // Total over the accepted strict-ceiling payload weights.
    amrex::Real total_weight = amrex::Real(0.0);
    for (std::size_t j = 0; j < weights.size(); ++j) {
        total_weight += static_cast<amrex::Real>(weights[j]);
    }

    amrex::Vector<amrex::Vector<amrex::ParticleReal>> attr_real;
    attr_real.push_back(weights);
    int const nattr_real = append_rrea_previous_position_attributes(
        species, x, y, z, px, py, pz, attr_real);
    amrex::Vector<amrex::Vector<int>> attr_int;
    species.AddNParticles(
        0,
        static_cast<long>(x.size()),
        x,
        y,
        z,
        px,
        py,
        pz,
        nattr_real,
        attr_real,
        0,
        attr_int,
        1);

    created_count += static_cast<std::uint64_t>(x.size());
    created_weight += total_weight;
    // Native seed count refresh is deferred to
    // diag/checkpoint writes (see RefreshNativeParticleCounts call sites).
}

void RreaWarpXCoupling::CreateLocalKineticElectrons(
    WarpX& warpx,
    std::vector<amrex::Real> const& x_m,
    std::vector<amrex::Real> const& y_m,
    std::vector<amrex::Real> const& z_m,
    std::vector<amrex::Real> const& ux,
    std::vector<amrex::Real> const& uy,
    std::vector<amrex::Real> const& uz,
    std::vector<amrex::Real> const& weight,
    std::vector<amrex::Real> const& kinetic_energy_eV,
    int step,
    bool population_ceiling_preflighted)
{
    CommitLocalParticleBatch(
        warpx, m_seed_species_name, m_max_electron_macros,
        "electron_secondary_batch", /*photon_momentum=*/false,
        m_kinetic_secondary_electron_count, m_kinetic_secondary_weight,
        x_m, y_m, z_m, ux, uy, uz, weight, kinetic_energy_eV,
        step, population_ceiling_preflighted);
}

void RreaWarpXCoupling::CreateLocalKineticPositrons(
    WarpX& warpx,
    std::vector<amrex::Real> const& x_m,
    std::vector<amrex::Real> const& y_m,
    std::vector<amrex::Real> const& z_m,
    std::vector<amrex::Real> const& ux,
    std::vector<amrex::Real> const& uy,
    std::vector<amrex::Real> const& uz,
    std::vector<amrex::Real> const& weight,
    std::vector<amrex::Real> const& kinetic_energy_eV,
    int step,
    bool population_ceiling_preflighted)
{
    CommitLocalParticleBatch(
        warpx, m_positron_species_name, m_max_positron_macros,
        "positron_secondary_batch", /*photon_momentum=*/false,
        m_positron_created_count, m_positron_created_weight,
        x_m, y_m, z_m, ux, uy, uz, weight, kinetic_energy_eV,
        step, population_ceiling_preflighted);
}

void RreaWarpXCoupling::CreateLocalPhotons(
    WarpX& warpx,
    std::vector<amrex::Real> const& x_m,
    std::vector<amrex::Real> const& y_m,
    std::vector<amrex::Real> const& z_m,
    std::vector<amrex::Real> const& ux,
    std::vector<amrex::Real> const& uy,
    std::vector<amrex::Real> const& uz,
    std::vector<amrex::Real> const& weight,
    std::vector<amrex::Real> const& photon_energy_eV,
    int step,
    bool population_ceiling_preflighted)
{
    CommitLocalParticleBatch(
        warpx, m_photon_species_name, m_max_photon_macros,
        "photon_secondary_batch", /*photon_momentum=*/true,
        m_photon_created_count, m_photon_created_weight,
        x_m, y_m, z_m, ux, uy, uz, weight, photon_energy_eV,
        step, population_ceiling_preflighted);
}

void RreaWarpXCoupling::AccumulateIonizationEvent(
    int lev,
    amrex::Real r_m,
    amrex::Real z_m,
    amrex::Real event_weight,
    amrex::Real secondary_energy_eV,
    amrex::Real energy_loss_eV)
{
    if (event_weight <= amrex::Real(0.0)) {
        return;
    }
    if (!m_events) {
        amrex::Abort(
            "RREA ionization source cannot be committed because the event accumulator is unavailable");
    }
    if (!EventSourcePositionInDomain(lev, r_m, z_m)) {
        amrex::Abort(
            "RREA ionization source position is outside the half-open physical domain");
    }

    m_events->AddPositiveIonSource(lev, r_m, z_m, event_weight);
    m_ion_pair_source_weight += event_weight;
    m_ion_pair_source_z_weighted_m += event_weight * z_m;
    if (secondary_energy_eV < m_low_energy_cutoff_eV) {
        m_events->AddLowElectronSource(lev, r_m, z_m, event_weight);
    } else {
        // This above-cutoff secondary belongs to the kinetic path; adding it to
        // ne_low as well would double-count its charge and energy.
    }
    m_events->AddIonizationDiagnostic(lev, r_m, z_m, event_weight, energy_loss_eV);
}

void RreaWarpXCoupling::AccumulatePositiveIonSourceOnly(
    int lev,
    amrex::Real r_m,
    amrex::Real z_m,
    amrex::Real ion_weight,
    amrex::Real energy_loss_eV)
{
    if (ion_weight <= amrex::Real(0.0)) {
        return;
    }
    if (!m_events) {
        amrex::Abort(
            "RREA positive-ion source cannot be committed because the event accumulator is unavailable");
    }
    if (!EventSourcePositionInDomain(lev, r_m, z_m)) {
        amrex::Abort(
            "RREA positive-ion source position is outside the half-open physical domain");
    }
    m_events->AddPositiveIonSource(lev, r_m, z_m, ion_weight);
    m_ion_pair_source_weight += ion_weight;
    m_ion_pair_source_z_weighted_m += ion_weight * z_m;
    m_events->AddIonizationDiagnostic(lev, r_m, z_m, ion_weight, energy_loss_eV);
}

void RreaWarpXCoupling::AccumulateLowEnergyElectron(
    int lev,
    amrex::Real r_m,
    amrex::Real z_m,
    amrex::Real electron_weight,
    amrex::Real energy_loss_eV)
{
    if (electron_weight <= amrex::Real(0.0)) {
        return;
    }
    if (!m_events) {
        amrex::Abort(
            "RREA low-energy-electron source cannot be committed because the event accumulator is unavailable");
    }
    if (!EventSourcePositionInDomain(lev, r_m, z_m)) {
        amrex::Abort(
            "RREA low-energy-electron source position is outside the half-open physical domain");
    }
    m_events->AddLowElectronSource(lev, r_m, z_m, electron_weight);
    m_events->AddIonizationDiagnostic(
        lev,
        r_m,
        z_m,
        electron_weight,
        amrex::max(energy_loss_eV, amrex::Real(0.0)));
}

void RreaWarpXCoupling::RecordPhotonLocalDeposit(
    amrex::Real event_weight,
    amrex::Real photon_energy_eV)
{
    if (event_weight > amrex::Real(0.0) && photon_energy_eV > amrex::Real(0.0)) {
        m_ledger.photon_local_deposit_energy_eV += event_weight * photon_energy_eV;
    }
}

void RreaWarpXCoupling::RecordPositronAnnihilation(
    amrex::Real event_weight,
    amrex::Real local_deposit_energy_eV,
    amrex::Real photon1_energy_eV,
    amrex::Real photon2_energy_eV,
    bool at_rest_or_below_cutoff)
{
    if (event_weight <= amrex::Real(0.0)) {
        return;
    }
    ++m_positron_annihilation_count;
    if (at_rest_or_below_cutoff) {
        ++m_positron_annihilation_at_rest_count;
    } else {
        ++m_positron_annihilation_in_flight_count;
    }
    m_positron_annihilation_weight += event_weight;
    m_positron_annihilation_photon_energy_eV += event_weight
        * (amrex::max(photon1_energy_eV, amrex::Real(0.0))
           + amrex::max(photon2_energy_eV, amrex::Real(0.0)));
    m_positron_annihilation_local_deposit_eV += event_weight
        * amrex::max(local_deposit_energy_eV, amrex::Real(0.0));
    // Photons created by annihilation enter the transport source flow in full.
    // If either falls below the transport cutoff, its corresponding local
    // deposit must therefore appear on the sink side exactly once.
    RecordPhotonLocalDeposit(event_weight, local_deposit_energy_eV);
}

}  // namespace rrea::warpx
