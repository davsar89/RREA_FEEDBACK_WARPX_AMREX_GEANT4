#ifndef RREA_TRANSPORT_STEP_H
#define RREA_TRANSPORT_STEP_H

// One RREA transport step: the per-step context every species phase shares,
// and the primitives they share with it.  RreaSpeciesTransport.cpp defines the
// electron, photon and positron phases (one tile loop) and
// RreaSecondaryCommit.cpp the secondary commit; RreaParticleInteraction.cpp
// keeps only the hook that builds the context and runs the four in order.
//
// What lives here is exactly what more than one phase needs: the captured
// pre-push component indices and the KineticStep every phase hands its
// particles to.  The particle physics itself is one history per particle kind
// in RreaKineticStep.H, shared by live parents and newborns.
//
// Pinned by rrea_charged_interleave_smoke (the kinematics primitives),
// rrea_kinetic_engine_smoke (the histories), rrea_secondary_group_smoke (the
// deposit and group contracts) and rrea_warpx_transport_apply (the assembled
// step).

#include "RreaKineticStep.H"
#include "RreaChargedFinalStates.H"
#include "RreaChargedInterleave.H"
#include "RreaMaterialPath.H"
#include "RreaParticleInteraction.H"
#include "RreaPathFieldWork.H"
#include "RreaPhotonFinalStates.H"
#include "RreaRng.H"
#include "RreaTransportLog.H"
#include "RreaWarpXCoupling.H"

#include "rrea/RreaDebugOptions.H"
#include "rrea/RreaInteractionTables.H"
#include "rrea/RreaLorentzPush.H"
#include "rrea/RreaSeedSchedule.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rrea::warpx {

struct PreviousRzComponentIndices {
    int r = -1;
    int z = -1;
    int theta = -1;
    int ux = -1;
    int uy = -1;
    int uz = -1;
};

inline PreviousRzComponentIndices require_previous_rz_components(
    WarpXParticleContainer& particles,
    std::string const& species_name)
{
    for (char const* component : {
             "rrea_prev_r", "rrea_prev_z", "rrea_prev_theta",
             "rrea_prev_ux", "rrea_prev_uy", "rrea_prev_uz"}) {
        if (!particles.HasRealComp(component)) {
            amrex::Abort(
                "RREA pre-boundary transport requires runtime component '"
                + std::string(component) + "' on species '" + species_name
                + "'; CaptureRreaPositionsBeforePush must run before WarpX::OneStep");
        }
    }
    return PreviousRzComponentIndices{
        particles.GetRealCompIndex("rrea_prev_r"),
        particles.GetRealCompIndex("rrea_prev_z"),
        particles.GetRealCompIndex("rrea_prev_theta"),
        particles.GetRealCompIndex("rrea_prev_ux"),
        particles.GetRealCompIndex("rrea_prev_uy"),
        particles.GetRealCompIndex("rrea_prev_uz")};
}

class ScopedMultipleMFIters {
public:
    ScopedMultipleMFIters()
        : m_previous(amrex::MFIter::allowMultipleMFIters(1))
    {}

    ~ScopedMultipleMFIters()
    {
        amrex::MFIter::allowMultipleMFIters(m_previous);
    }

    ScopedMultipleMFIters(ScopedMultipleMFIters const&) = delete;
    ScopedMultipleMFIters& operator=(ScopedMultipleMFIters const&) = delete;

private:
    int m_previous = 0;
};
// Per-step context shared by the transport phases.
struct TransportStep {
    WarpX& warpx;
    RreaWarpXCoupling& coupling;
    rrea::RreaInteractionTables const& tables;
    RreaParticleInteractionConfig const& m_config;
    int step;
    amrex::Real time_s;
    amrex::Real dt_s;
    amrex::Geometry const& geom;

        // All material/fluid/ledger mutations remain logged until the global
        // projected post-removal population has passed the collective ceiling
        // preflight.  This makes the batch fail closed without committing the
        // non-particle portion of any correlated interaction group first.
        TransportSideEffects pending_side_effects{};

        bool invalidated_particles = false;
        bool invalidated_photons = false;
        bool repositioned_photons = false;
        bool invalidated_positrons = false;
        // Set when a history writes a survivor position back.
        bool repositioned_electrons = false;
        bool repositioned_positrons = false;

    [[nodiscard]] KineticStep Kinetic() const {
        return {tables.View(),coupling.KineticView(),m_config,step,time_s,dt_s,
            {{geom.ProbLo(0),geom.ProbLo(1)},{geom.ProbHi(0),geom.ProbHi(1)}}};
    }
    // One tile loop for every species (RreaSpeciesTransport.cpp).
    template <class History>
    void TransportSpecies(std::string const& species, bool& invalidated,
                          bool& repositioned, History history);
    void ApplyElectronTransport();
    void ApplyPhotonTransport();
    void ApplyPositronTransport();
    void DrainAndCommitSecondaries();
};
} // namespace rrea::warpx
#endif
