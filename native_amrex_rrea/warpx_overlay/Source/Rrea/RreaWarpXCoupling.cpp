// The RREA field step: level bindings and the cell-centred field adapters,
// the pre-push position snapshot, the net-chord current deposit, and the
// BeforeFieldSolve hook that drives one fluid + Maxwell advance.

#include "RreaWarpXCoupling.H"

#include "RreaCouplingDetail.H"
#include "RreaParticleSweep.H"

#include "rrea/RreaFieldHandoff.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>

#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rrea::warpx {

namespace detail {

amrex::Real photon_species_energy(std::string const& name)
{
    if (name.empty()) {
        return amrex::Real(0.0);
    }
    amrex::Real local = amrex::Real(0.0);
    RreaForEachValidParticle(
        WarpX::GetInstance().GetPartContainer().GetParticleContainerFromName(name),
        [&local](RreaLiveParticle const& p) {
            local += p.weight * p.PhotonEnergyEv();
        });
    amrex::ParallelDescriptor::ReduceRealSum(local);
    return local;
}

}  // namespace detail

RreaWarpXCoupling& RreaWarpXCoupling::GetInstance()
{
    static RreaWarpXCoupling coupling;
    return coupling;
}

bool RreaWarpXCoupling::Enabled() const noexcept
{
    return m_enabled;
}

amrex::Real RreaWarpXCoupling::ProfileAltitudeMslAtZ(amrex::Real z_m) const noexcept
{
    return m_profile_altitude_msl_at_z0_m + z_m;
}

amrex::Real RreaWarpXCoupling::TransportReferenceDensityKgM3() const
{
    return m_interaction_tables.ReferenceDensityKgM3();
}

amrex::Real RreaWarpXCoupling::TransportDensityRatioAtZ(amrex::Real z_m) const
{
    // The deck's scalar is already expressed against the table density, so it
    // needs no conversion -- and must not require one, because table-free
    // coupling smokes run in exactly this mode.
    if (!m_transport_density_profile_enabled) {
        return m_transport_density_ratio;
    }
    amrex::Real const reference = TransportReferenceDensityKgM3();
    if (!(reference > amrex::Real(0.0))) {
        amrex::Abort(
            "rrea.air_density_model=altitude_profile supplies absolute density "
            "and cannot be scaled without the transport configuration reference_density_kg_m3");
    }
    return m_transport_density_profile.Interpolate(ProfileAltitudeMslAtZ(z_m))
        / reference;
}

amrex::Real RreaWarpXCoupling::MobilityDensityRatioAtZ(amrex::Real z_m) const
{
    // A reduced mobility is defined at Loschmidt, so re-express the transport
    // ratio against that instead of the table density.  Without tables there is
    // no reference to re-express against and the scalar ratio is the only
    // density statement the deck makes; that is the table-free smoke path.
    amrex::Real const transport_ratio = TransportDensityRatioAtZ(z_m);
    amrex::Real const reference = TransportReferenceDensityKgM3();
    if (!(reference > amrex::Real(0.0))) {
        return transport_ratio;
    }
    return transport_ratio * reference / kReducedMobilityReferenceDensityKgM3;
}

void RreaWarpXCoupling::EnsurePreviousPositionComponents(WarpX& warpx)
{
    if (m_previous_position_components_ready) {
        return;
    }
#if !defined(WARPX_DIM_RZ)
    amrex::Abort("RREA schema-6 position capture currently requires WarpX RZ");
#else
    auto ensure = [](WarpXParticleContainer& species, char const* name) {
        auto const names = species.GetRealSoANames();
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            species.AddRealComp(name, true);
        }
    };
    auto add_for_species = [&](std::string const& species_name) {
        auto& species =
            warpx.GetPartContainer().GetParticleContainerFromName(species_name);
        int do_resampling = 0;
        amrex::ParmParse species_pp(species_name);
        species_pp.query("do_resampling", do_resampling);
        if (do_resampling != 0) {
            amrex::Abort(
                "WarpX resampling is unsupported for RREA-managed species; "
                "use rrea.population_ceiling_policy (strict_abort_v1 or "
                "adaptive_resample_v1)");
        }
        ensure(species, "rrea_prev_r");
        ensure(species, "rrea_prev_z");
        ensure(species, "rrea_prev_theta");
        ensure(species, "rrea_prev_ux");
        ensure(species, "rrea_prev_uy");
        ensure(species, "rrea_prev_uz");
    };
    add_for_species(m_seed_species_name);
    add_for_species(m_photon_species_name);
    add_for_species(m_positron_species_name);
    m_previous_position_components_ready = true;
#endif
}

void RreaWarpXCoupling::CaptureRreaPositionsBeforePush(WarpX& warpx, int step)
{
    ReadParameters();
    if (!m_enabled) {
        return;
    }
    InitFromWarpX(warpx);
    EnsurePreviousPositionComponents(warpx);
#if defined(WARPX_DIM_RZ)
    auto capture = [&](std::string const& species_name) {
        auto& species =
            warpx.GetPartContainer().GetParticleContainerFromName(species_name);
        for (WarpXParIter pti(species, 0); pti.isValid(); ++pti) {
            auto& attribs = pti.GetAttribs();
            auto* const r = attribs[PIdx::r].dataPtr();
            auto* const z = attribs[PIdx::z].dataPtr();
            auto* const theta = attribs[PIdx::theta].dataPtr();
            auto* const ux = attribs[PIdx::ux].dataPtr();
            auto* const uy = attribs[PIdx::uy].dataPtr();
            auto* const uz = attribs[PIdx::uz].dataPtr();
            auto* const prev_r = pti.GetAttribs("rrea_prev_r").dataPtr();
            auto* const prev_z = pti.GetAttribs("rrea_prev_z").dataPtr();
            auto* const prev_theta = pti.GetAttribs("rrea_prev_theta").dataPtr();
            auto* const prev_ux = pti.GetAttribs("rrea_prev_ux").dataPtr();
            auto* const prev_uy = pti.GetAttribs("rrea_prev_uy").dataPtr();
            auto* const prev_uz = pti.GetAttribs("rrea_prev_uz").dataPtr();
            long const np = pti.numParticles();
            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long i) noexcept {
                    prev_r[i] = r[i];
                    prev_z[i] = z[i];
                    prev_theta[i] = theta[i];
                    prev_ux[i] = ux[i];
                    prev_uy[i] = uy[i];
                    prev_uz[i] = uz[i];
                });
        }
        amrex::Gpu::streamSynchronize();
    };
    capture(m_seed_species_name);
    capture(m_photon_species_name);
    capture(m_positron_species_name);
    if (m_maxwell_continuity_reprime_required) {
        // InjectScheduledSeeds runs immediately before this hook.  Deposit the
        // actual post-injection population on the same cell-centered adapter
        // used at the end of the step, then make it rho^n for the fail-closed
        // Maxwell material-continuity identity.  This extra deposit occurs
        // only on initialization/injection steps.
        DepositParticleChargeToRreaGrid(warpx);
        m_advance->PrimeMaxwellContinuityReference();
        m_maxwell_continuity_reprime_required = false;
    }
#endif
    m_positions_captured_for_step = true;
    m_positions_captured_step = step;
    m_transport_completed_for_step = false;
}

void RreaWarpXCoupling::AfterParticlePushBeforeBoundaries(
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
    if (dt_s <= amrex::Real(0.0)) {
        amrex::Abort("RREA post-push transport requires positive dt");
    }
    if (!m_positions_captured_for_step || m_positions_captured_step != step) {
        amrex::Abort(
            "RREA post-push transport did not find a matching pre-push position snapshot");
    }
    RefreshDensityRatioFields();
    ResetNetChordStep(warpx, dt_s);
    if (!m_debug_disable_transport && m_air_interaction) {
        ABLASTR_PROFILE("RreaWarpXCoupling::PostPushTransport");
        if (m_debug_actual_continuity_fixture) {
            RunActualContinuityFixtureStep(warpx, step);
            std::vector<RreaSecondaryParticle> no_candidates;
            std::array<std::vector<std::uint64_t>, 3> no_removed_keys;
            std::array<amrex::Long, 3> no_removals{0, 0, 0};
            PreflightAndApplySecondaryPopulationControl(
                warpx, no_candidates, no_removed_keys, no_removals,
                "actual_continuity_fixture", step);
        } else {
            m_air_interaction->Apply(
                warpx, *this, InteractionTables(), step, time_s, dt_s);
        }
        // adaptive_resample_v1 population control runs on the committed
        // post-transport population, before BeforeFieldSolve deposits
        // charge; its null-space moves conserve that deposited charge in
        // every realization.
        ApplyAdaptivePopulationControl(warpx, step);
    }
    m_transport_completed_for_step = true;
    m_transport_completed_step = step;
}

void RreaWarpXCoupling::BuildBindings(WarpX& warpx)
{
    using ablastr::fields::Direction;
    using ::warpx::fields::FieldType;

    if (warpx.finestLevel() > 0) {
        amrex::Abort("RREA coupling currently requires amr.max_level = 0");
    }

    auto& fields = warpx.GetMultiFabRegister();
    m_level_bindings.clear();
    if (!fields.has(FieldType::rho_fp, 0)
        || !fields.has(FieldType::Efield_fp, Direction{0}, 0)
        || !fields.has(FieldType::Efield_fp, Direction{2}, 0)) {
        amrex::Abort("RREA coupling requires rho_fp, Er, and Ez fields");
    }
    auto* rho = fields.get(FieldType::rho_fp, 0);
    m_native_er = fields.get(FieldType::Efield_fp, Direction{0}, 0);
    m_native_ez = fields.get(FieldType::Efield_fp, Direction{2}, 0);

    // The solver-side grids must be genuinely cell-centered, and WarpX's
    // rho_fp is nodal: its nodal-tagged BoxArray has boxes that overlap at
    // shared node rows, so event deposits on a shared row would land in only
    // one of two diverging fluid copies, volume integrals would double-count
    // those rows, and cell-centered-only operators would receive a nodal
    // layout.  convert(ba, cell) recovers the underlying non-overlapping cell
    // boxes (same count/order/DistributionMap, so fab-index alignment with the
    // native fields is preserved).
    amrex::BoxArray const cell_ba =
        amrex::convert(rho->boxArray(), amrex::IntVect::TheCellVector());
    m_rho_adapter = std::make_unique<amrex::MultiFab>(
        cell_ba, rho->DistributionMap(), 1, rho->nGrowVect());
    m_er_adapter = std::make_unique<amrex::MultiFab>(
        cell_ba, rho->DistributionMap(), 1, 0);
    m_ez_adapter = std::make_unique<amrex::MultiFab>(
        cell_ba, rho->DistributionMap(), 1, 0);
    m_rho_adapter->setVal(0.0);
    m_er_adapter->setVal(0.0);
    m_ez_adapter->setVal(0.0);
    m_density_ratio_field = std::make_unique<amrex::MultiFab>(
        cell_ba, rho->DistributionMap(), 1, 0);
    m_density_ratio_field_refreshed = false;

    m_level_bindings.push_back(RreaAmrexLevelBinding{
        warpx.Geom(0),
        cell_ba,
        rho->DistributionMap(),
        m_rho_adapter.get(),
        m_er_adapter.get(),
        m_ez_adapter.get(),
        m_density_ratio_field.get()});
    AuditLevelLayouts(0);
    RefreshDensityRatioFields();
    if (m_cd_plane_flux.Enabled()) {
#if (AMREX_SPACEDIM < 2)
        amrex::Abort("RREA C&D plane diagnostics require an RZ build");
#else
        auto const& geom = m_level_bindings.front().geom;
        if (geom.isPeriodic(1)) {
            amrex::Abort(
                "RREA C&D plane diagnostics reject periodic z because wrapped "
                "plane crossings have ambiguous signed-flux semantics");
        }
        if (m_cd_plane_core_radius_m > geom.ProbHi(0)
            || m_cd_plane_core_radius_m <= geom.ProbLo(0)) {
            amrex::Abort(
                "rrea.cd_plane_core_radius_m must lie inside the radial domain");
        }
        for (amrex::Real const plane_z_m : m_cd_plane_flux.PlaneZ()) {
            if (!(plane_z_m > geom.ProbLo(1) && plane_z_m < geom.ProbHi(1))) {
                amrex::Abort(
                    "every rrea.cd_plane_z_m location must lie strictly inside "
                    "the non-periodic axial domain");
            }
        }
#endif
    }
}

void RreaWarpXCoupling::RefreshDensityRatioFields()
{
    // The transport density ratio is a static
    // z-profile, so re-evaluating it over the whole mesh every step is pure
    // waste.  Refresh once per (re)binding; m_density_ratio_field_refreshed
    // is cleared wherever the level bindings are rebuilt.
    if (m_density_ratio_field_refreshed || !m_density_ratio_field) {
        return;
    }
    auto const& geom = m_level_bindings.front().geom;
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const domain_lo = geom.Domain().smallEnd();
    for (amrex::MFIter mfi(*m_density_ratio_field, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const arr = m_density_ratio_field->array(mfi);
        amrex::LoopOnCpu(mfi.tilebox(), [&](int i, int j, int k) noexcept {
            amrex::ignore_unused(i, k);
#if (AMREX_SPACEDIM >= 2)
            amrex::Real const z_m = plo[1]
                + (static_cast<amrex::Real>(j - domain_lo[1])
                   + amrex::Real(0.5)) * dx[1];
#else
            amrex::ignore_unused(j);
            amrex::Real const z_m = amrex::Real(0.0);
#endif
            arr(i, j, k) = MobilityDensityRatioAtZ(z_m);
        });
    }
    m_density_ratio_field_refreshed = true;
}

void RreaWarpXCoupling::AuditLevelLayouts(int lev) const
{
    auto const& binding = m_level_bindings.at(lev);
    AMREX_ALWAYS_ASSERT(binding.rho_c_per_m3 != nullptr);
    AMREX_ALWAYS_ASSERT(binding.efield_x != nullptr);
    AMREX_ALWAYS_ASSERT(binding.efield_z != nullptr);
    AMREX_ALWAYS_ASSERT(binding.rho_c_per_m3->nComp() == 1);
    AMREX_ALWAYS_ASSERT(binding.efield_x->nComp() == 1);
    AMREX_ALWAYS_ASSERT(binding.efield_z->nComp() == 1);
}

void RreaWarpXCoupling::DepositParticleChargeToRreaGrid(WarpX& warpx)
{
    ablastr::fields::MultiLevelScalarField rho_cell;
    rho_cell.push_back(m_rho_adapter.get());
    warpx.GetPartContainer().DepositCharge(rho_cell, amrex::Real(0.0));
    ablastr::utils::communication::SumBoundary(
        *m_rho_adapter,
        0,
        m_rho_adapter->nComp(),
        m_rho_adapter->nGrowVect(),
        m_rho_adapter->nGrowVect(),
        WarpX::do_single_precision_comms,
        m_level_bindings.front().geom.periodicity());
}

void RreaWarpXCoupling::CopyAdaptersToWarpXFields()
{
    // staggering_aware_handoff: cell-center -> native staggering (averaging
    // along every native NODE dim, one-sided at physical edges), so the field
    // WarpX gathers sits at the location its deposition/gather kernels assume.
    auto const& geom = m_level_bindings.front().geom;
    rrea::InterpolateCellCenteredToNative(*m_er_adapter, *m_native_er, geom);
    rrea::InterpolateCellCenteredToNative(*m_ez_adapter, *m_native_ez, geom);
}

bool RreaWarpXCoupling::PrepareCdElectronPlaneCrossings(
    std::vector<RreaPlaneFluxCrossing> const& crossings,
    RreaPlaneFluxAccumulator::PreparedUpdate& prepared) const
{
    if (!m_cd_plane_flux.Enabled()) {
        return crossings.empty();
    }
    return m_cd_plane_flux.Accumulator().PrepareCrossings(crossings, prepared);
}

void RreaWarpXCoupling::CommitPreparedCdElectronPlaneCrossings(
    RreaPlaneFluxAccumulator::PreparedUpdate&& prepared) noexcept
{
    m_cd_plane_flux.Accumulator().CommitPrepared(std::move(prepared));
}

void RreaWarpXCoupling::ResetNetChordStep(WarpX& warpx, amrex::Real dt_s)
{
    auto const& geom = warpx.Geom(0);
    if (geom.isPeriodic(1)) {
        amrex::Abort(
            "RREA net-chord current deposition does not support periodic z: "
            "a wrapped endpoint pair would deposit a segment nobody flew");
    }
    int constexpr pad = 2;
    auto const& domain = geom.Domain();
    m_netchord_physical_nr = domain.length(0);
    m_netchord_physical_nz = domain.length(1);
    m_netchord_physical_z_begin = pad;
    m_netchord_grid.nr = m_netchord_physical_nr + pad;
    m_netchord_grid.nz = m_netchord_physical_nz + 2 * pad;
    m_netchord_grid.dr = geom.CellSize(0);
    m_netchord_grid.dz = geom.CellSize(1);
    m_netchord_grid.r_lo = geom.ProbLo(0);
    m_netchord_grid.z_lo = geom.ProbLo(1) - pad * geom.CellSize(1);
    m_netchord_grid.resize();
    m_netchord_dt_s = dt_s;
    m_netchord_segment_count = 0;
    m_netchord_z_displacement_e_m = amrex::Real(0.0);
    m_netchord_created_charge_e = amrex::Real(0.0);
    m_netchord_escaped_charge_e = amrex::Real(0.0);
    m_netchord_demoted_charge_e = amrex::Real(0.0);
    m_netchord_annihilated_charge_e = amrex::Real(0.0);
}

void RreaWarpXCoupling::AccumulateChargedSegment(
    int lev,
    amrex::Real r0_m,
    amrex::Real z0_m,
    amrex::Real r1_m,
    amrex::Real z1_m,
    amrex::Real charge_e_weight,
    std::uint64_t channels)
{
    if (lev != 0) {
        amrex::Abort(
            "RREA net-chord deposition is single-level (pinned mode contract)");
    }
    if (channels & kRreaSegmentChannelCreated) {
        m_netchord_created_charge_e += charge_e_weight;
    }
    if (channels & kRreaSegmentChannelEscaped) {
        m_netchord_escaped_charge_e += charge_e_weight;
    }
    if (channels & kRreaSegmentChannelDemoted) {
        m_netchord_demoted_charge_e += charge_e_weight;
    }
    if (channels & kRreaSegmentChannelAnnihilated) {
        m_netchord_annihilated_charge_e += charge_e_weight;
    }
    if (charge_e_weight == amrex::Real(0.0)) {
        return;
    }
    if (!(m_netchord_dt_s > amrex::Real(0.0)) || m_netchord_grid.rho.empty()) {
        amrex::Abort(
            "RREA net-chord segment arrived before the per-step scratch was armed");
    }
    ++m_netchord_segment_count;
    m_netchord_z_displacement_e_m += charge_e_weight * (z1_m - z0_m);
    double const q = static_cast<double>(charge_e_weight);
    rrea::DepositSegmentRZ(
        m_netchord_grid, q, static_cast<double>(m_netchord_dt_s),
        r0_m, z0_m, r1_m, z1_m);
}

void RreaWarpXCoupling::AccumulateChargedEscape(
    int lev,
    amrex::Real trajectory_start_r_m,
    amrex::Real trajectory_start_z_m,
    amrex::Real final_microchord_start_r_m,
    amrex::Real final_microchord_start_z_m,
    amrex::Real crossing_r_m,
    amrex::Real crossing_z_m,
    amrex::Real charge_e_weight,
    std::uint64_t channels,
    std::uint32_t faces)
{
    if (lev != 0) {
        amrex::Abort(
            "RREA charged escape deposition is single-level (pinned mode contract)");
    }
    if ((channels & kRreaSegmentChannelEscaped) == 0U) {
        amrex::Abort(
            "RREA charged escape deposition is missing its escaped channel");
    }
    if (channels & kRreaSegmentChannelCreated) {
        m_netchord_created_charge_e += charge_e_weight;
    }
    m_netchord_escaped_charge_e += charge_e_weight;
    if (charge_e_weight == amrex::Real(0.0)) {
        return;
    }
    if (!(m_netchord_dt_s > amrex::Real(0.0))
        || m_netchord_grid.rho.empty()
        || m_netchord_physical_nr <= 0
        || m_netchord_physical_nz <= 0) {
        amrex::Abort(
            "RREA charged escape arrived before the per-step scratch was armed");
    }
    double const q = static_cast<double>(charge_e_weight);
    auto const deposit = rrea::DepositEscapingSegmentRZ(
        m_netchord_grid,
        q,
        static_cast<double>(m_netchord_dt_s),
        trajectory_start_r_m,
        trajectory_start_z_m,
        final_microchord_start_r_m,
        final_microchord_start_z_m,
        crossing_r_m,
        crossing_z_m,
        m_netchord_physical_nr,
        m_netchord_physical_z_begin,
        m_netchord_physical_nz,
        faces);
    if (!deposit.valid) {
        amrex::Abort(
            "RREA charged escape could not close its exact boundary CIC cloud");
    }
    ++m_netchord_segment_count;
    m_netchord_z_displacement_e_m += charge_e_weight
        * (crossing_z_m - trajectory_start_z_m)
        + static_cast<amrex::Real>(
            deposit.explicit_z_displacement_charge);
}

amrex::Real RreaWarpXCoupling::LowEnergyCutoffEv() const noexcept
{
    return m_low_energy_cutoff_eV;
}

RreaInteractionTables const& RreaWarpXCoupling::InteractionTables() const noexcept
{
    return m_interaction_tables;
}

RreaFieldSample RreaWarpXCoupling::GatherFieldRZ(
    int lev,
    amrex::Real r_m,
    amrex::Real z_m) const
{
    // The engine owns the total field on the replicated Maxwell vectors, so
    // the gather is a direct evaluation valid on every rank: no scratch halo,
    // no rank-local FAB ownership, no post-push redistribution contract.
    if (m_advance == nullptr) {
        amrex::Abort("RREA field gather has no bound engine advance");
    }
    (void)lev;  // single level: the Maxwell state is one replicated mesh
    double er = 0.0;
    double ez = 0.0;
    double btheta = 0.0;
    m_advance->GatherTotalFieldRZ(r_m, z_m, er, ez, btheta);
    return RreaFieldSample{
        static_cast<amrex::Real>(er),
        static_cast<amrex::Real>(ez),
        static_cast<amrex::Real>(btheta)};
}

bool RreaWarpXCoupling::EventSourcePositionInDomain(
    int lev,
    amrex::Real r_m,
    amrex::Real z_m) const noexcept
{
    if (lev < 0 || lev >= static_cast<int>(m_level_bindings.size())) {
        return false;
    }
    auto const prob_lo = m_level_bindings[lev].geom.ProbLoArray();
    auto const prob_hi = m_level_bindings[lev].geom.ProbHiArray();
    return RreaRzPositionInHalfOpenDomain(
        r_m, z_m,
        prob_lo[0], prob_hi[0], prob_lo[1], prob_hi[1]);
}

bool RreaWarpXCoupling::SecondaryGroupPositionInDomain(
    int lev,
    amrex::Real r_m,
    amrex::Real z_m) const noexcept
{
    return EventSourcePositionInDomain(lev, r_m, z_m);
}

void RreaWarpXCoupling::RefreshNativeParticleCounts(WarpX& warpx)
{
    if (!m_enabled) {
        return;
    }
    auto& electrons = warpx.GetPartContainer().GetParticleContainerFromName(m_seed_species_name);
    m_native_seed_particle_count =
        static_cast<std::uint64_t>(electrons.TotalNumberOfParticles());
}

bool RreaWarpXCoupling::BeforeFieldSolve(
    WarpX& warpx,
    amrex::Real dt_s,
    int step,
    amrex::Real time_s,
    SpaceChargeSolvePurpose purpose)
{
    ABLASTR_PROFILE("RreaWarpXCoupling::BeforeFieldSolve");
    ReadParameters();
    if (!m_enabled) {
        return false;
    }
    InitFromWarpX(warpx);
    RefreshDensityRatioFields();
    bool const initialization = purpose == SpaceChargeSolvePurpose::Initialization;
    if (!initialization
        && (!m_transport_completed_for_step || m_transport_completed_step + 1 != step)) {
        amrex::Abort(
            "RREA evolution field solve did not follow exactly one completed post-push transport epoch");
    }

    if (!initialization) {
        ++m_before_field_solve_count;
        ++m_field_advance_count;
    }

    if (!initialization && dt_s <= amrex::Real(0.0)) {
        amrex::Abort("RREA coupling requires positive dt");
    }
    bool const reduced_diag_due =
        !initialization
        && !m_debug_disable_diagnostics
        && m_diag_interval > 0
        && step % m_diag_interval == 0;
    // Log the first confirmation and the reduced-diagnostic cadence.
    if (m_before_field_solve_count == 1 || reduced_diag_due) {
        amrex::Print() << "RREA Maxwell field advance: step="
                       << step
                       << ", time_s=" << time_s
                       << ", dt_s=" << dt_s
                       << ", hook_call_count=" << m_before_field_solve_count
                       << ", field_advance_count=" << m_field_advance_count
                       << "\n";
    }

    // A restored initialization keeps the checkpointed rho adapter: the
    // initialization callback runs before the first restarted push, and a
    // fresh deposit here would replace that exact old-time state.  The
    // first evolution solve deposits the post-push charge normally.
    if (!(initialization && m_restored_from_checkpoint)) {
        // WarpX's own rho_fp is left undeposited: this hook replaces the
        // electrostatic solve, so nothing reads it.  The engine's charge is
        // deposited straight onto the cell-centred RREA grid, after post-push
        // transport for evolution and once for a new-run static initialization.
        ABLASTR_PROFILE("RreaWarpXCoupling::FieldHandoff");
        DepositParticleChargeToRreaGrid(warpx);
    }

    if (initialization) {
        // Fresh initialization and checkpoint restore both have the actual
        // particle adapter and persisted fluid state bound here. Scheduled
        // injection, if any, re-primes once more in the pre-push hook.
        m_advance->PrimeMaxwellContinuityReference();
        m_maxwell_continuity_reprime_required = false;
        if (!m_restored_from_checkpoint && !m_debug_disable_field_copyback) {
            // The run starts on the analytic ambient (scattered field zero)
            // and seed charge builds its field dynamically at finite speed.
            CopyAdaptersToWarpXFields();
        }
        if (!m_debug_disable_field_copyback) {
            // Refresh physical field guards on both fresh initialization and
            // restored native state so the first gather is independent of
            // checkpointed ghost bytes.
            warpx.ApplyEfieldBoundary(0, PatchType::fine, time_s);
        }
        return true;
    }

    {
        ABLASTR_PROFILE("RreaWarpXCoupling::EventSync");
        m_events->NormalizeAndSync(
            0,
            dt_s,
            /*normalize_diagnostic_fields=*/reduced_diag_due,
            /*fill_boundaries=*/true);
        auto& fluid = m_advance->Fluid(0);
        amrex::MultiFab::Copy(
            fluid.LowElectronSourceRate(),
            m_events->LowElectronSourceRate(0), 0, 0, 1, 0);
        amrex::MultiFab::Copy(
            fluid.PositiveIonSourceRate(),
            m_events->PositiveIonSourceRate(0), 0, 0, 1, 0);
        amrex::MultiFab::Copy(
            fluid.DirectNegativeIonSourceRate(),
            m_events->DirectNegativeIonSourceRate(0), 0, 0, 1, 0);
    }

    m_advance_config.collect_ion_drift_center_diagnostics = reduced_diag_due;
    // Sampled after this step's transport and population control, so a macro
    // the resample just boosted anchors the gate that is about to judge it.
    m_advance_config.maximum_represented_charge_e = static_cast<double>(
        detail::maximum_represented_charge_e(
            {m_seed_species_name, m_photon_species_name,
             m_positron_species_name}));
    m_advance->BindKineticCurrent(
        &m_netchord_grid.jr, &m_netchord_grid.jz,
        /*pad_cells=*/2, /*charge_per_index_flux_c=*/rrea::qe);
    m_advance->SetMaxwellBoundaryRecordDir(m_output_dir);
    {
        ABLASTR_PROFILE("RreaWarpXCoupling::FluidAndMaxwell");
        m_advance->AdvanceOneStep(dt_s, m_advance_config);
    }
    m_advance->AppendMaxwellBoundaryRecord(
        static_cast<double>(time_s), static_cast<double>(dt_s));
    if (m_debug_actual_continuity_fixture) {
        VerifyActualContinuityFixtureAfterAdvance(warpx, step);
    }

    if (!m_debug_disable_field_copyback) {
        ABLASTR_PROFILE("RreaWarpXCoupling::FieldCopyback");
        CopyAdaptersToWarpXFields();
        // The RREA hook returns before WarpX's electrostatic solver can run its
        // normal boundary path.  Apply the configured physical E-field boundary
        // conditions after the valid nodal/Yee values have been synchronized
        // and the interior/periodic guards filled.
        warpx.ApplyEfieldBoundary(0, PatchType::fine, time_s);
    }

    if (!initialization && m_cd_plane_flux.Enabled()) {
        m_cd_plane_last_observed_step = step;
        m_cd_plane_last_observed_time_s = time_s;
        MaybeWriteCdPlaneFlux(step, time_s, /*force=*/false);
    }

    if (m_video_diag_enable) {
        ABLASTR_PROFILE("RreaWarpXCoupling::VideoFrame");
        MaybeWriteVideoFrame(warpx, step, time_s);
    }

    if (reduced_diag_due) {
        ABLASTR_PROFILE("RreaWarpXCoupling::WriteDiagnostics");
        // Refresh the diagnostic-only native seed
        // count here (and in WriteCheckpoint) instead of after every spawn.
        // adaptive_resample_v1 may thin or split BETWEEN the spawns and this
        // point, which is exactly why the count is re-read here rather than
        // cached per spawn: the refresh sees the committed post-resample
        // population, and the reduced CSV's macro-count telemetry sawtooths
        // with the roulette by design (weight sums do not).
        RefreshNativeParticleCounts(warpx);
        WriteDiagnostics(m_output_dir, step, time_s);
    }

    m_events->Reset(0);
    return true;
}


}  // namespace rrea::warpx
