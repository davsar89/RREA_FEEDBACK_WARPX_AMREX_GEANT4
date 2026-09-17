// Population control for RreaWarpXCoupling: the strict ceiling backstop and
// the adaptive_resample_v1 CIC-conserving thinning/splitting transaction.

#include "RreaWarpXCoupling.H"

#include "RreaCouplingDetail.H"
#include "RreaParticleInteraction.H"
#include "RreaParticleSweep.H"
#include "RreaPopulationGpu.H"
#include "RreaPopulationRecord.H"
#include "RreaPopulationParticles.H"
#include "RreaRng.H"

#include "rrea/RreaCicConservingResample.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace rrea::warpx {

using detail::append_rrea_previous_position_attributes;

namespace {


template <typename T>
std::vector<T> alltoall_trivial_records(
    std::vector<std::vector<T>> const& send)
{
    static_assert(std::is_trivially_copyable_v<T>);
    int const ranks = amrex::ParallelDescriptor::NProcs();
    if (send.size() != static_cast<std::size_t>(ranks)) {
        amrex::Abort("RREA all-to-all record exchange has the wrong rank count");
    }
#ifdef AMREX_USE_MPI
    std::vector<int> send_bytes(ranks, 0), receive_bytes(ranks, 0);
    std::vector<int> send_offsets(ranks, 0), receive_offsets(ranks, 0);
    std::size_t send_total = 0;
    for (int p = 0; p < ranks; ++p) {
        if (send[p].size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max()) / sizeof(T)
            || send_total > static_cast<std::size_t>(
                std::numeric_limits<int>::max())
                - send[p].size() * sizeof(T)) {
            amrex::Abort("RREA all-to-all send payload exceeds MPI int range");
        }
        send_offsets[p] = static_cast<int>(send_total);
        send_bytes[p] = static_cast<int>(send[p].size() * sizeof(T));
        send_total += static_cast<std::size_t>(send_bytes[p]);
    }
    std::vector<T> packed(send_total / sizeof(T));
    std::size_t cursor = 0;
    for (auto const& partition : send) {
        std::copy(partition.begin(), partition.end(), packed.begin() + cursor);
        cursor += partition.size();
    }
    MPI_Comm const comm = amrex::ParallelDescriptor::Communicator();
    MPI_Alltoall(
        send_bytes.data(), 1, MPI_INT,
        receive_bytes.data(), 1, MPI_INT, comm);
    std::size_t receive_total = 0;
    for (int p = 0; p < ranks; ++p) {
        if (receive_bytes[p] < 0
            || receive_total > static_cast<std::size_t>(
                std::numeric_limits<int>::max() - receive_bytes[p])) {
            amrex::Abort("RREA all-to-all receive payload exceeds MPI int range");
        }
        receive_offsets[p] = static_cast<int>(receive_total);
        receive_total += static_cast<std::size_t>(receive_bytes[p]);
    }
    if (receive_total % sizeof(T) != 0) {
        amrex::Abort("RREA all-to-all received a malformed record payload");
    }
    std::vector<T> received(receive_total / sizeof(T));
    MPI_Alltoallv(
        packed.data(), send_bytes.data(), send_offsets.data(), MPI_BYTE,
        received.data(), receive_bytes.data(), receive_offsets.data(), MPI_BYTE,
        comm);
    return received;
#else
    return send.empty() ? std::vector<T>{} : send.front();
#endif
}

}  // namespace

void RreaWarpXCoupling::EnforcePopulationCeiling(
    WarpX& warpx,
    std::string const& species_name,
    long ceiling,
    amrex::Long global_candidates,
    char const* origin,
    int step,
    amrex::Long global_removals) const
{
    if (ceiling == 0) {
        return;
    }
    auto& species =
        warpx.GetPartContainer().GetParticleContainerFromName(species_name);
    // Terminal interaction parents are marked invalid during assembly and are
    // therefore already absent here.  global_removals contains only still-valid
    // parents scheduled for absorbing-boundary deletion after this hook.
    amrex::Long const current = species.TotalNumberOfParticles(true, false);
    amrex::Long const cap = static_cast<amrex::Long>(ceiling);
    if (current < 0 || global_candidates < 0 || global_removals < 0
        || global_removals > current) {
        amrex::Abort(
            "RREA secondary transaction reported an invalid parent-removal count for "
            + species_name);
    }
    std::uint64_t projected = 0;
    bool const fits = RreaStrictPopulationProjection(
        static_cast<std::uint64_t>(current),
        static_cast<std::uint64_t>(global_removals),
        static_cast<std::uint64_t>(global_candidates),
        static_cast<std::uint64_t>(cap),
        projected,
        /*reject_equality=*/m_cd_plane_flux.Enabled());
    if (fits) {
        return;
    }

    std::ostringstream message;
    message << "RREA strict population ceiling "
            << (m_cd_plane_flux.Enabled() ? "reached or exceeded" : "exceeded")
            << " for " << species_name
            << " at step " << step << " (origin=" << (origin ? origin : "unknown")
            << ", current=" << current << ", candidates=" << global_candidates
            << ", removals=" << global_removals
            << ", projected=" << projected << ", ceiling=" << cap << ")";
    amrex::Abort(message.str());
}

void RreaWarpXCoupling::PreflightSecondaryBatch(
    WarpX& warpx,
    amrex::Long local_electron_candidates,
    amrex::Long local_photon_candidates,
    amrex::Long local_positron_candidates,
    amrex::Long local_electron_removals,
    amrex::Long local_photon_removals,
    amrex::Long local_positron_removals,
    char const* origin,
    int step) const
{
    std::array<amrex::Long, 6> global = {
        local_electron_candidates,
        local_photon_candidates,
        local_positron_candidates,
        local_electron_removals,
        local_photon_removals,
        local_positron_removals};
    amrex::ParallelDescriptor::ReduceLongSum(global.data(), global.size());
    EnforcePopulationCeiling(
        warpx, m_seed_species_name, m_max_electron_macros, global[0], origin, step, global[3]);
    EnforcePopulationCeiling(
        warpx, m_photon_species_name, m_max_photon_macros, global[1], origin, step, global[4]);
    EnforcePopulationCeiling(
        warpx, m_positron_species_name, m_max_positron_macros, global[2], origin, step, global[5]);
}

void RreaWarpXCoupling::PreflightAndApplySecondaryPopulationControl(
    WarpX& warpx,
    std::vector<RreaSecondaryParticle>& candidates,
    std::array<std::vector<std::uint64_t>, 3> const& removed_parent_keys,
    std::array<amrex::Long, 3> const& local_parent_removals,
    char const* origin,
    int step)
{
#if defined(WARPX_DIM_RZ)
    struct LiveSlot {
        std::uint64_t id = 0;
        std::uint64_t* idcpu = nullptr;
        amrex::ParticleReal* weight = nullptr;
        amrex::Real energy_eV = 0.0;
        amrex::ParticleReal original_weight = 0.0;
        amrex::ParticleReal planned_weight = 0.0;
    };
    struct CandidateSlot {
        std::uint64_t high = 0;
        std::uint64_t low = 0;
        std::size_t index = 0;
        amrex::ParticleReal planned_weight = 0.0;
    };
    struct ChargedPlan {
        std::vector<LiveSlot> live;
        std::vector<CandidateSlot> staged;
        rrea::RreaGlobalCicPlan global;
        amrex::Long local_live_removed = 0;
        amrex::Long local_staged_removed = 0;
        amrex::Real killed_weight = 0.0;
        amrex::Real boosted_weight = 0.0;
        amrex::Real killed_energy_eV = 0.0;
        amrex::Real boosted_energy_eV = 0.0;
        bool thinning_due = false;
    };
    struct SupportSummary {
        int support_r = 0;
        int support_z = 0;
        int owner_rank = 0;
        std::uint64_t count = 0;
        std::uint64_t capacity = 0;
        std::uint64_t floor = 0;
        std::uint64_t request = 0;
    };

    auto const& geom = warpx.Geom(0);
    auto const domain = geom.Domain();
    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const dx = geom.CellSizeArray();
    int const rank = amrex::ParallelDescriptor::MyProc();
    int const ranks = amrex::ParallelDescriptor::NProcs();
    PopulationRecordBuilder record;
    for (int d=0;d<2;++d) {
        record.plo[d]=plo[d];record.phi[d]=phi[d];record.dx[d]=dx[d];
        record.small[d]=domain.smallEnd(d);record.periodic[d]=geom.isPeriodic(d);
    }
    record.rank=rank;
    auto build_plan = [&](std::string const& species_name,
                          RreaSecondarySpecies secondary_species,
                          long target, std::size_t removal_index,
                          std::size_t diagnostic_index,
                          RreaAdaptiveResampleSpecies rng_species,
                          unsigned preapplied_bit) {
        ChargedPlan result;
        if (target <= 0 || m_population_ceiling_policy != "adaptive_resample_v1"
            || m_population_control_interval <= 0
            || step % m_population_control_interval != 0) {
            return result;
        }
        std::vector<rrea::RreaGlobalCicParticle> local;
        auto const& pending_removals = removed_parent_keys[removal_index];
        rrea::GpuVector<std::uint64_t> sorted_removals(
            pending_removals.begin(),pending_removals.end());
        std::sort(sorted_removals.begin(),sorted_removals.end());
        BuildPopulationLiveRecord const build_live{record,
            {sorted_removals.data(),sorted_removals.size()}};
        RreaMapLiveParticles<PopulationLiveRecord>(
            warpx.GetPartContainer().GetParticleContainerFromName(species_name),
            KineticGpuEnabled(),build_live,[&](PopulationLiveRecord const& p) {
                local.push_back(p.particle);
                auto const weight=static_cast<amrex::ParticleReal>(p.particle.weight);
                result.live.push_back(LiveSlot{p.particle.stable_id,p.idcpu,p.weight,
                    p.energy,weight,weight});
            });
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            auto const& candidate = candidates[i];
            if (candidate.species != secondary_species) continue;
            std::uint64_t const low =
                (candidate.transient_lineage << 2)
                | static_cast<std::uint64_t>(
                    static_cast<int>(candidate.root_species));
            double const u = rrea::RreaElectronProperVelocityFromKineticEv(
                candidate.kinetic_or_photon_energy_eV);
            double const theta = std::atan2(candidate.y_m, candidate.x_m);
            local.push_back(record(
                std::hypot(candidate.x_m, candidate.y_m), candidate.z_m,
                u * candidate.dir_x, u * candidate.dir_y, u * candidate.dir_z,
                theta, candidate.weight, true,
                candidate.root_rng_particle_id, low));
            result.staged.push_back(CandidateSlot{
                candidate.root_rng_particle_id, low, i,
                static_cast<amrex::ParticleReal>(candidate.weight)});
        }
        std::uint64_t const stream_key =
            rng_species == RreaAdaptiveResampleSpecies::Electron
                ? 0x656c656374726f6eULL : 0x706f736974726f6eULL;
        std::vector<std::vector<rrea::RreaGlobalCicParticle>> to_owner(ranks);
        for (auto const& particle : local) {
            int const owner = rrea::RreaGlobalCicSupportOwner(
                particle.support_r, particle.support_z, ranks, stream_key);
            to_owner[owner].push_back(particle);
        }
        auto owned = alltoall_trivial_records(to_owner);
        if (local.size() > static_cast<std::size_t>(
                std::numeric_limits<amrex::Long>::max() / ranks)) {
            amrex::Abort("adaptive_resample_v1 global particle count can overflow Long");
        }
        amrex::Long global_count = static_cast<amrex::Long>(local.size());
        amrex::ParallelDescriptor::ReduceLongSum(global_count);
        std::size_t const target_count = static_cast<std::size_t>(target);
        if (target_count > std::numeric_limits<std::size_t>::max()
                - target_count / 4U) {
            amrex::Abort("adaptive_resample_v1 target overflows its control band");
        }
        result.thinning_due = static_cast<std::size_t>(global_count)
            > target_count + target_count / 4U;
        std::sort(owned.begin(), owned.end(), [](auto const& lhs, auto const& rhs) {
            if (lhs.support_r != rhs.support_r) return lhs.support_r < rhs.support_r;
            if (lhs.support_z != rhs.support_z) return lhs.support_z < rhs.support_z;
            if (lhs.phase != rhs.phase) return lhs.phase < rhs.phase;
            if (lhs.candidate != rhs.candidate) return lhs.candidate < rhs.candidate;
            if (lhs.stable_id_high != rhs.stable_id_high) {
                return lhs.stable_id_high < rhs.stable_id_high;
            }
            return lhs.stable_id < rhs.stable_id;
        });
        struct OwnerGroup {
            std::size_t begin = 0;
            std::size_t end = 0;
            SupportSummary summary;
        };
        std::vector<OwnerGroup> groups;
        amrex::Long owner_capacity = 0;
        amrex::Long owner_floor = 0;
        for (std::size_t begin = 0; begin < owned.size();) {
            std::size_t end = begin + 1U;
            while (end < owned.size()
                   && owned[end].support_r == owned[begin].support_r
                   && owned[end].support_z == owned[begin].support_z) ++end;
            // Topology/capacity remains a host extended-precision reference.
            // GPU plans must agree with it before any weights are committed.
            auto const support=rrea::KineticSpan<rrea::RreaGlobalCicParticle const>{
                owned.data()+begin,end-begin};
            std::size_t const capacity=rrea::RreaCicResampleRemovalCapacity(support);
            std::size_t const floor=support.size()-capacity;
            if (capacity > static_cast<std::size_t>(
                    std::numeric_limits<amrex::Long>::max() - owner_capacity)
                || floor > static_cast<std::size_t>(
                    std::numeric_limits<amrex::Long>::max() - owner_floor)) {
                amrex::Abort("adaptive_resample_v1 owner capacity overflows Long");
            }
            owner_capacity += static_cast<amrex::Long>(capacity);
            owner_floor += static_cast<amrex::Long>(floor);
            groups.push_back(OwnerGroup{begin, end, SupportSummary{
                owned[begin].support_r, owned[begin].support_z, rank,
                static_cast<std::uint64_t>(support.size()),
                static_cast<std::uint64_t>(capacity),
                static_cast<std::uint64_t>(floor), 0}});
            begin = end;
        }
        amrex::Long global_capacity = owner_capacity;
        amrex::Long global_floor = owner_floor;
        amrex::ParallelDescriptor::ReduceLongSum(global_capacity);
        amrex::ParallelDescriptor::ReduceLongSum(global_floor);
        result.global.capacity = static_cast<std::size_t>(global_capacity);
        result.global.geometric_floor = static_cast<std::size_t>(global_floor);
        m_resample_exact_capacity[diagnostic_index] =
            global_capacity;
        m_resample_geometric_floor[diagnostic_index] =
            global_floor;
        if (result.thinning_due) {
            m_population_control_preapplied_step = step;
            m_population_control_preapplied_species |= preapplied_bit;
        }
        amrex::Long nominal_request = result.thinning_due
            ? std::min(
                global_count - static_cast<amrex::Long>(target),
                global_capacity)
            : 0;
        result.global.requested = static_cast<std::size_t>(nominal_request);

        int const root = amrex::ParallelDescriptor::IOProcessorNumber();
        std::vector<std::vector<SupportSummary>> summaries_to_root(ranks);
        for (auto const& group : groups) {
            summaries_to_root[root].push_back(group.summary);
        }
        auto global_summaries = alltoall_trivial_records(summaries_to_root);
        std::vector<std::vector<SupportSummary>> requests_to_owner(ranks);
        if (rank == root) {
            std::sort(
                global_summaries.begin(), global_summaries.end(),
                [](auto const& lhs, auto const& rhs) {
                    if (lhs.support_r != rhs.support_r) {
                        return lhs.support_r < rhs.support_r;
                    }
                    return lhs.support_z < rhs.support_z;
                });
            amrex::Long remaining = nominal_request;
            std::uint64_t summary_count = 0, summary_capacity = 0,
                          summary_floor = 0;
            for (std::size_t i = 0; i < global_summaries.size(); ++i) {
                auto summary = global_summaries[i];
                if ((i > 0
                     && summary.support_r == global_summaries[i - 1].support_r
                     && summary.support_z == global_summaries[i - 1].support_z)
                    || summary.owner_rank != rrea::RreaGlobalCicSupportOwner(
                        summary.support_r, summary.support_z, ranks, stream_key)) {
                    amrex::Abort("adaptive_resample_v1 support ownership is inconsistent");
                }
                summary_count += summary.count;
                summary_capacity += summary.capacity;
                summary_floor += summary.floor;
                summary.request = static_cast<std::uint64_t>(std::min<amrex::Long>(
                    remaining, static_cast<amrex::Long>(summary.capacity)));
                remaining -= static_cast<amrex::Long>(summary.request);
                requests_to_owner[summary.owner_rank].push_back(summary);
            }
            if (summary_count != static_cast<std::uint64_t>(global_count)
                || summary_capacity != static_cast<std::uint64_t>(global_capacity)
                || summary_floor != static_cast<std::uint64_t>(global_floor)
                || remaining != 0) {
                amrex::Abort("adaptive_resample_v1 support summaries do not close globally");
            }
        }
        auto requests = alltoall_trivial_records(requests_to_owner);
        std::map<std::pair<int, int>, SupportSummary> request_by_support;
        for (auto const& request : requests) {
            if (request.owner_rank != rank
                || !request_by_support.emplace(
                    std::make_pair(request.support_r, request.support_z),
                    request).second) {
                amrex::Abort("adaptive_resample_v1 routed an invalid support request");
            }
        }
        if (request_by_support.size() != groups.size()) {
            amrex::Abort("adaptive_resample_v1 did not return every owner support");
        }

        std::vector<std::vector<rrea::RreaGlobalCicUpdate>> updates_to_source(ranks);
        amrex::Long owner_removed = 0;
        std::vector<PopulationSupportRequest> support_requests;
        support_requests.reserve(groups.size());
        for (auto const& group : groups) {
            auto const found=request_by_support.find(std::make_pair(
                group.summary.support_r,group.summary.support_z));
            support_requests.push_back({group.begin,group.end,
                static_cast<std::size_t>(found->second.request),
                static_cast<std::size_t>(group.summary.capacity),
                static_cast<std::size_t>(group.summary.floor)});
        }
        PopulationRandomStream population_rng{m_rng_seed,
            m_population_controller_rng_salt,static_cast<std::uint64_t>(step),rng_species};
        ExecutePopulationSupportPlans(owned,support_requests,ranks,stream_key,population_rng,
            [&](PopulationPlanResult const& plan,
                rrea::KineticSpan<rrea::RreaGlobalCicUpdate const> updates) {
                if(plan.removed>static_cast<std::size_t>(
                    std::numeric_limits<amrex::Long>::max()-owner_removed))
                    amrex::Abort("adaptive_resample_v1 removal count overflow");
                owner_removed+=static_cast<amrex::Long>(plan.removed);
                for(auto const& update:updates)
                    updates_to_source[update.source_rank].push_back(update);
            });
        amrex::Long planned_removed = owner_removed;
        amrex::ParallelDescriptor::ReduceLongSum(planned_removed);
        result.global.removed = static_cast<std::size_t>(planned_removed);
        result.global.updates = alltoall_trivial_records(updates_to_source);
        if (nominal_request > 0 && planned_removed < nominal_request) {
            amrex::Abort("adaptive_resample_v1 owner plans made insufficient progress");
        }
        using Identity = std::tuple<bool, std::uint64_t, std::uint64_t>;
        std::map<Identity, amrex::ParticleReal> updates;
        for (auto const& update : result.global.updates) {
            if (update.source_rank != rank) {
                amrex::Abort("adaptive_resample_v1 misrouted a particle update");
            }
            auto const inserted = updates.emplace(
                Identity{update.candidate, update.stable_id_high,
                         update.stable_id},
                static_cast<amrex::ParticleReal>(update.stored_weight));
            if (!inserted.second) {
                amrex::Abort("adaptive_resample_v1 produced a duplicate local update");
            }
        }
        auto account = [&](double old_weight, double new_weight,
                           amrex::Real energy_eV) {
            if (new_weight < old_weight) {
                amrex::Real const delta = old_weight - new_weight;
                result.killed_weight += delta;
                result.killed_energy_eV += delta * energy_eV;
            } else if (new_weight > old_weight) {
                amrex::Real const delta = new_weight - old_weight;
                result.boosted_weight += delta;
                result.boosted_energy_eV += delta * energy_eV;
            }
        };
        for (auto& slot : result.live) {
            auto const found = updates.find(Identity{false, 0, slot.id});
            if (found != updates.end()) {
                slot.planned_weight = found->second;
                updates.erase(found);
            }
            account(slot.original_weight, slot.planned_weight, slot.energy_eV);
            result.local_live_removed += slot.planned_weight == 0.0 ? 1 : 0;
        }
        for (auto& slot : result.staged) {
            auto const found = updates.find(Identity{true, slot.high, slot.low});
            if (found != updates.end()) {
                slot.planned_weight = found->second;
                updates.erase(found);
            }
            auto const& candidate = candidates[slot.index];
            account(candidate.weight, slot.planned_weight,
                    candidate.kinetic_or_photon_energy_eV);
            result.local_staged_removed += slot.planned_weight == 0.0 ? 1 : 0;
        }
        if (!updates.empty()) {
            amrex::Abort("adaptive_resample_v1 received an unknown local update");
        }
        amrex::Long routed_removed =
            result.local_live_removed + result.local_staged_removed;
        amrex::ParallelDescriptor::ReduceLongSum(routed_removed);
        if (routed_removed != static_cast<amrex::Long>(result.global.removed)) {
            amrex::Abort("adaptive_resample_v1 global removal routing is inconsistent");
        }
        return result;
    };

    m_population_control_preapplied_step = step;
    m_population_control_preapplied_species = 0;
    ChargedPlan electron = build_plan(
        m_seed_species_name, RreaSecondarySpecies::Electron,
        m_population_target_electron_macros, 0, 0,
        RreaAdaptiveResampleSpecies::Electron, 1U);
    ChargedPlan positron = build_plan(
        m_positron_species_name, RreaSecondarySpecies::Positron,
        m_population_target_positron_macros, 2, 1,
        RreaAdaptiveResampleSpecies::Positron, 2U);

    std::array<amrex::Long, 3> local_candidates{0, 0, 0};
    for (auto const& candidate : candidates) {
        std::size_t const index = candidate.species == RreaSecondarySpecies::Electron
            ? 0U : candidate.species == RreaSecondarySpecies::Photon ? 1U : 2U;
        ++local_candidates[index];
    }
    local_candidates[0] -= static_cast<amrex::Long>(std::count_if(
        electron.staged.begin(), electron.staged.end(),
        [](auto const& slot) { return slot.planned_weight == 0.0; }));
    local_candidates[2] -= static_cast<amrex::Long>(std::count_if(
        positron.staged.begin(), positron.staged.end(),
        [](auto const& slot) { return slot.planned_weight == 0.0; }));
    if (local_parent_removals[0]
            > std::numeric_limits<amrex::Long>::max()
                - electron.local_live_removed
        || local_parent_removals[2]
            > std::numeric_limits<amrex::Long>::max()
                - positron.local_live_removed) {
        amrex::Abort("adaptive_resample_v1 removal count overflow");
    }
    PreflightSecondaryBatch(
        warpx, local_candidates[0], local_candidates[1], local_candidates[2],
        local_parent_removals[0] + electron.local_live_removed,
        local_parent_removals[1],
        local_parent_removals[2] + positron.local_live_removed,
        origin, step);

    auto apply = [&](ChargedPlan& plan,
                     WarpXParticleContainer& species) {
        bool invalidated=plan.local_live_removed>0;
#ifdef RREA_USE_CUDA
        if(KineticGpuEnabled()) {
            rrea::GpuVector<PopulationWeightCommit> live;
            live.reserve(plan.live.size());
            for (auto const& slot : plan.live) {
                live.push_back({slot.idcpu, slot.weight, slot.planned_weight});
            }
            rrea::GpuFor(GpuModule::Always, live.size(), ApplyPopulationWeightCommit{live.data()});
        } else
#endif
        {
            for(auto const& slot:plan.live) {
                *slot.weight=slot.planned_weight;
                if(slot.planned_weight==0.0) {
                    amrex::ParticleIDWrapper<> id(*slot.idcpu);id.make_invalid();
                }
            }
        }
        for (auto const& slot : plan.staged) {
            candidates[slot.index].weight = slot.planned_weight;
        }
        if (invalidated) species.deleteInvalidParticles();
        m_resample_killed_count += static_cast<std::uint64_t>(
            plan.local_live_removed + plan.local_staged_removed);
        m_resample_killed_weight += plan.killed_weight;
        m_resample_boost_weight += plan.boosted_weight;
        m_resample_killed_energy_eV += plan.killed_energy_eV;
        m_resample_boost_energy_eV += plan.boosted_energy_eV;
        m_resample_thin_rounds += plan.global.removed > 0 ? 1U : 0U;
    };
    auto& electron_species = warpx.GetPartContainer()
        .GetParticleContainerFromName(m_seed_species_name);
    auto& positron_species = warpx.GetPartContainer()
        .GetParticleContainerFromName(m_positron_species_name);
    apply(electron, electron_species);
    apply(positron, positron_species);
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(), [](auto const& p) {
            return p.species != RreaSecondarySpecies::Photon
                && p.weight == amrex::Real(0.0);
        }),
        candidates.end());
#else
    amrex::ignore_unused(
        warpx, candidates, removed_parent_keys, local_parent_removals,
        origin, step);
    amrex::Abort("adaptive_resample_v1 requires the WarpX RZ build");
#endif
}

void RreaWarpXCoupling::ApplyAdaptivePopulationControl(WarpX& warpx, int step)
{
    // Global thinning was preflighted with staged candidates inside the
    // secondary transaction. This step-boundary tail performs only split
    // refill; reaching an un-preflighted thin action is a broken invariant.
    if (m_population_ceiling_policy != "adaptive_resample_v1") {
        return;
    }
    if (m_population_control_interval <= 0
        || step % m_population_control_interval != 0) {
        return;
    }
    if (m_population_target_electron_macros > 0) {
        if (m_population_control_preapplied_step != step
            || (m_population_control_preapplied_species & 1U) == 0U) {
            AdaptiveResampleOneSpecies(
                warpx, m_seed_species_name, m_population_target_electron_macros);
        }
    }
    // No photon branch: the photon target is frozen at zero because photon
    // roulette would change the realized alive photon energy, which the
    // strict photon transport ledger does not account for.
    if (m_population_target_positron_macros > 0) {
        // Positron resample energy shares the electron counters; no strict
        // in-run charged-energy ledger gates it (the photon ledger's
        // annihilation flow is tallied at realization time, so killing a
        // positron macro before it annihilates stays consistent).
        // Electron and positron reductions own distinct append-only RNG
        // channels, so equal support/step keys do not correlate the species.
        if (m_population_control_preapplied_step != step
            || (m_population_control_preapplied_species & 2U) == 0U) {
            AdaptiveResampleOneSpecies(
                warpx, m_positron_species_name, m_population_target_positron_macros);
        }
    }
}

void RreaWarpXCoupling::AdaptiveResampleOneSpecies(
    WarpX& warpx,
    std::string const& species_name,
    long target)
{
#if defined(WARPX_DIM_RZ)
    auto& species =
        warpx.GetPartContainer().GetParticleContainerFromName(species_name);
    amrex::Long const count = species.TotalNumberOfParticles(true, false);
    double const split_min_w =
        static_cast<double>(m_population_split_min_weight);
    if (species_name != m_seed_species_name
        && species_name != m_positron_species_name) {
        amrex::Abort(
            "adaptive_resample_v1 received an unmanaged charged species");
    }
    if (RreaAdaptiveResampleAction(count, target, split_min_w)
        == RreaResampleAction::Thin) {
        amrex::Abort(
            "adaptive_resample_v1 thinning was not preflighted with staged candidates");
    }

    // ---------------- SPLIT: one doubling round ----------------
    if (RreaAdaptiveResampleAction(count, target, split_min_w)
        == RreaResampleAction::Split) {
        amrex::Vector<amrex::ParticleReal> cx;
        amrex::Vector<amrex::ParticleReal> cy;
        amrex::Vector<amrex::ParticleReal> cz;
        amrex::Vector<amrex::ParticleReal> cpx;
        amrex::Vector<amrex::ParticleReal> cpy;
        amrex::Vector<amrex::ParticleReal> cpz;
        amrex::Vector<amrex::ParticleReal> cw;
        RreaMapLiveParticles<PopulationSplitRecord>(species,KineticGpuEnabled(),
            SplitPopulationParticle{split_min_w},[&](PopulationSplitRecord const& child) {
                cx.push_back(child.x);cy.push_back(child.y);cz.push_back(child.z);
                cpx.push_back(child.ux);cpy.push_back(child.uy);cpz.push_back(child.uz);
                cw.push_back(child.weight);
            });
        amrex::Long global_clone_count =
            static_cast<amrex::Long>(cx.size());
        amrex::ParallelDescriptor::ReduceLongSum(global_clone_count);
        if (global_clone_count > 0) {
            // Exact, unbiased: each split moves half the parent weight to an
            // identical clone; Monte-Carlo transport decorrelates the pair.
            amrex::Real clone_w_total = amrex::Real(0.0);
            for (auto const w : cw) {
                clone_w_total += static_cast<amrex::Real>(w);
            }
            amrex::Vector<amrex::Vector<amrex::ParticleReal>> attr_real;
            attr_real.push_back(cw);
            int const nattr_real = append_rrea_previous_position_attributes(
                species, cx, cy, cz, cpx, cpy, cpz, attr_real);
            amrex::Vector<amrex::Vector<int>> attr_int;
            species.AddNParticles(
                0,
                static_cast<long>(cx.size()),
                cx, cy, cz, cpx, cpy, cpz,
                nattr_real, attr_real, 0, attr_int, 1);
            m_resample_split_count += static_cast<std::uint64_t>(cx.size());
            m_resample_split_weight += clone_w_total;
        }
    }
#else
    amrex::ignore_unused(warpx, species_name, target);
    amrex::Abort("adaptive_resample_v1 requires the WarpX RZ build");
#endif
}

}  // namespace rrea::warpx
