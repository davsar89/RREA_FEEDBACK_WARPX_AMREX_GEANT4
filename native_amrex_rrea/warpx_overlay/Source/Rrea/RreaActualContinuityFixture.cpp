// Charge-continuity fixture members of RreaWarpXCoupling. This debug fixture
// runs in the production WarpX executable but is never a physics substitute: it
// drives every charged lifecycle branch (survivor, demotion, annihilation,
// native and newborn escape, birth) through the production accumulators and
// population controller, and the engine's own Maxwell continuity gate
// (RreaAmrexAdvance) is the check.  The fixture asserts only that the
// controller split and thinned as the deck requires.

#include "RreaWarpXCoupling.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "RreaCouplingDetail.H"

namespace rrea::warpx {

void RreaWarpXCoupling::InjectActualContinuityFixtureParticles(
    WarpX& warpx,
    int step)
{
#if defined(WARPX_DIM_RZ)
    if (!m_debug_actual_continuity_fixture
        || m_actual_continuity_fixture_injected) {
        return;
    }
    if (m_actual_continuity_fixture_steps_completed != 0) {
        amrex::Abort(
            "actual-continuity fixture attempted its initial injection after evolution");
    }
    auto const& geom = warpx.Geom(0);
    auto const& domain = geom.Domain();
    if (domain.length(0) < 8 || domain.length(1) < 12) {
        amrex::Abort("actual-continuity fixture requires at least an 8x12 RZ mesh");
    }
    auto const plo = geom.ProbLoArray();
    auto const dx = geom.CellSizeArray();
    int const support_r = domain.smallEnd(0) + domain.length(0) / 2 - 1;
    int const support_z = domain.smallEnd(1) + domain.length(1) / 2 - 1;
    auto support_position = [&](double fr, double fz) {
        return std::pair<amrex::Real, amrex::Real>{
            static_cast<amrex::Real>(
                plo[0]
                + (static_cast<double>(support_r - domain.smallEnd(0))
                   + 0.5 + fr)
                    * dx[0]),
            static_cast<amrex::Real>(
                plo[1]
                + (static_cast<double>(support_z - domain.smallEnd(1))
                   + 0.5 + fz)
                    * dx[1])};
    };

    auto const e0 = support_position(0.20, 0.25);
    auto const e1 = support_position(0.80, 0.75);
    auto owned = [](std::initializer_list<amrex::Real> values) {
        return amrex::ParallelDescriptor::IOProcessor()
            ? std::vector<amrex::Real>(values) : std::vector<amrex::Real>{};
    };
    CreateLocalSecondaries(
        warpx, RreaSecondarySpecies::Electron,
        owned({e0.first, e1.first}), owned({0.0, 0.0}),
        owned({e0.second, e1.second}), owned({0.0, 0.0}),
        owned({0.0, 0.0}), owned({1.0, 1.0}), owned({4.0, 4.0}),
        owned({1.0e5, 1.0e5}), step);

    auto const p0 = support_position(0.30, 0.35);
    auto const p1 = support_position(0.70, 0.65);
    CreateLocalSecondaries(
        warpx, RreaSecondarySpecies::Positron,
        owned({p0.first, p1.first}), owned({0.0, 0.0}),
        owned({p0.second, p1.second}), owned({0.0, 0.0}),
        owned({0.0, 0.0}), owned({1.0, 1.0}), owned({3.0, 3.0}),
        owned({1.0e5, 1.0e5}), step);
    m_actual_continuity_fixture_injected = true;
    m_maxwell_continuity_reprime_required = true;
    amrex::Print()
        << "RREA actual-continuity fixture initialized: electrons=2, positrons=2\n";
#else
    amrex::ignore_unused(warpx, step);
    amrex::Abort("actual-continuity fixture requires WarpX RZ");
#endif
}

void RreaWarpXCoupling::RunActualContinuityFixtureStep(
    WarpX& warpx,
    int step)
{
#if defined(WARPX_DIM_RZ)
    if (!m_debug_actual_continuity_fixture
        || !m_actual_continuity_fixture_injected
        || m_actual_continuity_fixture_steps_completed > 1) {
        amrex::Abort("actual-continuity fixture entered an invalid evolution phase");
    }

    auto const& geom = warpx.Geom(0);
    auto const& domain = geom.Domain();
    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const dx = geom.CellSizeArray();
    int const support_r = domain.smallEnd(0) + domain.length(0) / 2 - 1;
    int const support_z = domain.smallEnd(1) + domain.length(1) / 2 - 1;
    auto support_position = [&](double fr, double fz) {
        return std::pair<amrex::Real, amrex::Real>{
            static_cast<amrex::Real>(
                plo[0]
                + (static_cast<double>(support_r - domain.smallEnd(0))
                   + 0.5 + fr)
                    * dx[0]),
            static_cast<amrex::Real>(
                plo[1]
                + (static_cast<double>(support_z - domain.smallEnd(1))
                   + 0.5 + fz)
                    * dx[1])};
    };

    struct ChargedSlot {
        std::uint64_t stable_id = 0;
        std::uint64_t* idcpu = nullptr;
        amrex::ParticleReal* r = nullptr;
        amrex::ParticleReal* z = nullptr;
        amrex::ParticleReal* theta = nullptr;
        amrex::ParticleReal* weight = nullptr;
        amrex::ParticleReal* previous_r = nullptr;
        amrex::ParticleReal* previous_z = nullptr;
    };
    auto collect = [](WarpXParticleContainer& species) {
        std::vector<ChargedSlot> slots;
        for (WarpXParIter pti(species, 0); pti.isValid(); ++pti) {
            auto& tile = pti.GetParticleTile();
            auto tile_data = tile.getParticleTileData();
            auto& attribs = pti.GetAttribs();
            auto* const r = attribs[PIdx::r].dataPtr();
            auto* const z = attribs[PIdx::z].dataPtr();
            auto* const theta = attribs[PIdx::theta].dataPtr();
            auto* const weight = attribs[PIdx::w].dataPtr();
            auto* const previous_r = pti.GetAttribs("rrea_prev_r").dataPtr();
            auto* const previous_z = pti.GetAttribs("rrea_prev_z").dataPtr();
            long const np = pti.numParticles();
            for (long ip = 0; ip < np; ++ip) {
                auto particle_id = tile.id(static_cast<int>(ip));
                if (!particle_id.is_valid()) {
                    continue;
                }
                auto& raw_idcpu = tile_data.idcpu(static_cast<int>(ip));
                slots.push_back(ChargedSlot{
                    static_cast<std::uint64_t>(raw_idcpu), &raw_idcpu,
                    &r[ip], &z[ip], &theta[ip], &weight[ip],
                    &previous_r[ip], &previous_z[ip]});
            }
        }
        std::sort(
            slots.begin(), slots.end(),
            [](ChargedSlot const& lhs, ChargedSlot const& rhs) {
                return lhs.stable_id < rhs.stable_id;
            });
        return slots;
    };
    auto global_ids = [](std::vector<ChargedSlot> const& slots) {
        std::vector<std::uint64_t> local;
        local.reserve(slots.size());
        for (auto const& slot : slots) local.push_back(slot.stable_id);
#ifdef AMREX_USE_MPI
        int const local_count = static_cast<int>(local.size());
        int const ranks = amrex::ParallelDescriptor::NProcs();
        std::vector<int> counts(ranks, 0), displacements(ranks, 0);
        MPI_Comm const comm = amrex::ParallelDescriptor::Communicator();
        MPI_Allgather(
            &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
        int total = 0;
        for (int p = 0; p < ranks; ++p) {
            displacements[p] = total;
            total += counts[p];
        }
        std::vector<std::uint64_t> global(static_cast<std::size_t>(total));
        MPI_Allgatherv(
            local.data(), local_count, MPI_UINT64_T,
            global.data(), counts.data(), displacements.data(),
            MPI_UINT64_T, comm);
#else
        auto global = local;
#endif
        std::sort(global.begin(), global.end());
        return global;
    };
    auto ordinal = [](std::vector<std::uint64_t> const& ids,
                      std::uint64_t id) {
        auto const found = std::lower_bound(ids.begin(), ids.end(), id);
        if (found == ids.end() || *found != id) {
            amrex::Abort("actual-continuity fixture lost a global particle id");
        }
        return static_cast<std::size_t>(found - ids.begin());
    };
    auto invalidate = [](ChargedSlot const& slot) {
        amrex::ParticleIDWrapper<> particle_id(*slot.idcpu);
        particle_id.make_invalid();
    };
    auto commit_position = [](ChargedSlot const& slot, amrex::Real r, amrex::Real z) {
        *slot.r = static_cast<amrex::ParticleReal>(r);
        *slot.z = static_cast<amrex::ParticleReal>(z);
        *slot.theta = amrex::ParticleReal(0.0);
    };

    auto& electrons = warpx.GetPartContainer().GetParticleContainerFromName(
        m_seed_species_name);
    auto& positrons = warpx.GetPartContainer().GetParticleContainerFromName(
        m_positron_species_name);
    auto electron_slots = collect(electrons);
    auto positron_slots = collect(positrons);
    auto const electron_ids = global_ids(electron_slots);
    auto const positron_ids = global_ids(positron_slots);

    if (m_actual_continuity_fixture_steps_completed == 0) {
        std::array<std::pair<amrex::Real, amrex::Real>, 2> const electron_end{
            support_position(0.30, 0.40), support_position(0.70, 0.60)};
        std::array<std::pair<amrex::Real, amrex::Real>, 2> const positron_end{
            support_position(0.35, 0.30), support_position(0.65, 0.70)};
        for (auto const& slot : electron_slots) {
            std::size_t const i = ordinal(electron_ids, slot.stable_id);
            AccumulateChargedSegment(
                *slot.previous_r, *slot.previous_z,
                electron_end[i].first, electron_end[i].second,
                -static_cast<amrex::Real>(*slot.weight), 0U);
            commit_position(slot, electron_end[i].first, electron_end[i].second);
        }
        for (auto const& slot : positron_slots) {
            std::size_t const i = ordinal(positron_ids, slot.stable_id);
            AccumulateChargedSegment(
                *slot.previous_r, *slot.previous_z,
                positron_end[i].first, positron_end[i].second,
                static_cast<amrex::Real>(*slot.weight), 0U);
            commit_position(slot, positron_end[i].first, positron_end[i].second);
        }
        electrons.Redistribute();
        positrons.Redistribute();
        return;
    }

    // Two existing electrons survive and land on the same raw four-cell support as
    // the kinetic births below, so the real production CIC thinner can act.
    std::array<std::pair<amrex::Real, amrex::Real>, 2> const survivor_end{
        support_position(0.16, 0.22), support_position(0.88, 0.78)};
    for (auto const& slot : electron_slots) {
        std::size_t const i = ordinal(electron_ids, slot.stable_id);
        if (i < 2) {
            AccumulateChargedSegment(
                *slot.previous_r, *slot.previous_z,
                survivor_end[i].first, survivor_end[i].second,
                -static_cast<amrex::Real>(*slot.weight), 0U);
            commit_position(slot, survivor_end[i].first, survivor_end[i].second);
        } else if (i == 2) {
            auto const point = support_position(0.50, 0.50);
            amrex::Real const weight = *slot.weight;
            AccumulateLowEnergyElectron(point.first, point.second, weight);
            AccumulateChargedSegment(
                *slot.previous_r, *slot.previous_z,
                point.first, point.second, -weight,
                kRreaSegmentChannelDemoted);
            invalidate(slot);
        } else {
            amrex::Real const weight = *slot.weight;
            AccumulateChargedEscape(
                *slot.previous_r, *slot.previous_z,
                support_position(0.45, 0.90).first,
                static_cast<amrex::Real>(phi[1] - 0.20 * dx[1]),
                support_position(0.45, 0.90).first,
                static_cast<amrex::Real>(phi[1]),
                -weight, kRreaSegmentChannelEscaped, rrea::kEscapeFaceZHi);
            invalidate(slot);
        }
    }
    // Redistribute owns both the explicit repositioning above and deletion of
    // the two invalid terminal particles, matching the production transport
    // commit path.
    electrons.Redistribute();

    for (auto const& slot : positron_slots) {
        std::size_t const i = ordinal(positron_ids, slot.stable_id);
        amrex::Real const weight = *slot.weight;
        if (i == 0) {
            auto const point = support_position(0.42, 0.36);
            AccumulatePositiveIonSourceOnly(point.first, point.second, weight);
            AccumulateChargedSegment(
                *slot.previous_r, *slot.previous_z,
                point.first, point.second, weight,
                kRreaSegmentChannelAnnihilated);
        } else {
            auto const z_at_exit = support_position(0.55, 0.64).second;
            AccumulateChargedEscape(
                *slot.previous_r, *slot.previous_z,
                static_cast<amrex::Real>(phi[0] - 0.20 * dx[0]), z_at_exit,
                static_cast<amrex::Real>(phi[0]), z_at_exit,
                weight, kRreaSegmentChannelEscaped, rrea::kEscapeFaceRHi);
        }
        invalidate(slot);
    }
    positrons.deleteInvalidParticles();

    auto const birth = support_position(0.50, 0.50);
    std::array<double, 8> const birth_fr{
        0.10, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82, 0.94};
    std::array<double, 8> const birth_fz{
        0.18, 0.67, 0.31, 0.82, 0.44, 0.73, 0.27, 0.56};
    std::array<amrex::Real, 8> const birth_weight{
        amrex::Real(1.1), amrex::Real(1.3), amrex::Real(1.7), amrex::Real(2.1),
        amrex::Real(2.6), amrex::Real(3.2), amrex::Real(4.1), amrex::Real(5.3)};
    std::vector<amrex::Real> x, y, z, ux, uy, uz, weight, energy;
    x.reserve(8); y.reserve(8); z.reserve(8);
    ux.reserve(8); uy.reserve(8); uz.reserve(8);
    weight.reserve(8); energy.reserve(8);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (std::size_t i = 0; i < birth_fr.size(); ++i) {
            auto const end = support_position(birth_fr[i], birth_fz[i]);
            AccumulatePositiveIonSourceOnly(
                birth.first, birth.second, birth_weight[i]);
            AccumulateChargedSegment(
                birth.first, birth.second, end.first, end.second,
                -birth_weight[i], kRreaSegmentChannelCreated);
            x.push_back(end.first);
            y.push_back(amrex::Real(0.0));
            z.push_back(end.second);
            ux.push_back(amrex::Real(0.0));
            uy.push_back(amrex::Real(0.0));
            uz.push_back(amrex::Real(1.0));
            weight.push_back(birth_weight[i]);
            energy.push_back(amrex::Real(1.0e5));
        }
    }
    CreateLocalSecondaries(
        warpx, RreaSecondarySpecies::Electron, x, y, z, ux, uy, uz, weight, energy, step);

    // A ninth paired birth exits during its residual leg and is never inserted
    // into the particle container.  This forces the transient-newborn version
    // of the same boundary-cloud continuity equation.
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Real const newborn_weight = amrex::Real(1.9);
        amrex::Real const newborn_r = birth.first;
        amrex::Real const newborn_z = static_cast<amrex::Real>(
            plo[1] + 0.25 * dx[1]);
        AccumulatePositiveIonSourceOnly(newborn_r, newborn_z, newborn_weight);
        AccumulateChargedEscape(
            newborn_r, newborn_z,
            newborn_r, static_cast<amrex::Real>(plo[1] + 0.10 * dx[1]),
            newborn_r, static_cast<amrex::Real>(plo[1]),
            -newborn_weight,
            kRreaSegmentChannelCreated | kRreaSegmentChannelEscaped,
            rrea::kEscapeFaceZLo);
    }
#else
    amrex::ignore_unused(warpx, step);
    amrex::Abort("actual-continuity fixture requires WarpX RZ");
#endif
}

void RreaWarpXCoupling::VerifyActualContinuityFixtureAfterAdvance(
    WarpX& warpx,
    int step)
{
    amrex::ignore_unused(step);
    auto& electrons = warpx.GetPartContainer().GetParticleContainerFromName(
        m_seed_species_name);
    auto& positrons = warpx.GetPartContainer().GetParticleContainerFromName(
        m_positron_species_name);
    amrex::Long const electron_count =
        electrons.TotalNumberOfParticles(true, false);
    amrex::Long const positron_count =
        positrons.TotalNumberOfParticles(true, false);
    amrex::Long global_split_count = static_cast<amrex::Long>(
        m_resample_split_count);
    amrex::Long global_killed_count = static_cast<amrex::Long>(
        m_resample_killed_count);
    amrex::ParallelDescriptor::ReduceLongSum(global_split_count);
    amrex::ParallelDescriptor::ReduceLongSum(global_killed_count);

    // Phase 0: two electrons above the split floor split once each.
    if (m_actual_continuity_fixture_steps_completed == 0) {
        if (electron_count != 4 || positron_count != 2
            || global_split_count != 2
            || m_resample_thin_rounds != 0) {
            amrex::Abort(
                "actual-continuity fixture phase 0 did not force survivor+split");
        }
        ++m_actual_continuity_fixture_steps_completed;
        return;
    }

    // Phase 1: ten electrons on one support against a target of six must be
    // thinned, and the support capacity splits into exact and floor parts.
    if (electron_count < 3 || electron_count > 7 || positron_count != 0
        || global_split_count != 2
        || m_resample_thin_rounds == 0 || global_killed_count == 0
        || m_resample_exact_capacity[0] <= 0
        || m_resample_geometric_floor[0] <= 0
        || m_resample_exact_capacity[0] + m_resample_geometric_floor[0] != 10) {
        amrex::Abort(
            "actual-continuity fixture phase 1 missed a required lifecycle branch");
    }
    ++m_actual_continuity_fixture_steps_completed;
    amrex::Print()
        << "RREA_ACTUAL_CONTINUITY_FIXTURE_PASS"
        << " continuity="
        << m_advance->LastDiagnostics().maxwell_charge_continuity_max_relative
        << " final_electrons=" << electron_count
        << " thin_rounds=" << m_resample_thin_rounds << "\n";
}

}  // namespace rrea::warpx
