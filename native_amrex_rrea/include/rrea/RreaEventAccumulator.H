#ifndef RREA_EVENT_ACCUMULATOR_H
#define RREA_EVENT_ACCUMULATOR_H

#include <AMReX_BaseFab.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include "rrea/RreaConstants.H"

#include <cstdint>
#include <memory>
#include <vector>

namespace rrea {

// RZ shell volume of radial cell i, in the accumulator's index convention
// (index radial_domain_lo sits at r_problo).  rz_shell_volume owns the
// formula.
[[nodiscard]] AMREX_GPU_HOST_DEVICE inline double RreaRzShellCellVolume(
    double r_problo, double dr, double dz, int i, int radial_domain_lo) noexcept
{
    double const r_inner =
        r_problo + static_cast<double>(i - radial_domain_lo) * dr;
    return rz_shell_volume(r_inner, r_inner + dr, dz);
}

struct RreaEventAccumulatorLevelBinding {
    amrex::Geometry geom;
    amrex::BoxArray box_array;
    amrex::DistributionMapping distribution_map;
};

class RreaEventAccumulator {
public:
    RreaEventAccumulator(
        amrex::Vector<RreaEventAccumulatorLevelBinding> level_bindings,
        int n_grow);

    int FinestLevel() const noexcept;

    void Reset(int lev);
    void ResetAll();

    void AddLowElectronSource(int lev, double r_m, double z_m, double weight);
    void AddPositiveIonSource(int lev, double r_m, double z_m, double weight);
    void AddDirectNegativeIonSource(int lev, double r_m, double z_m, double weight);
    void AddIonizationDiagnostic(
        int lev,
        double r_m,
        double z_m,
        double weight,
        double energy_loss_eV);

    // The three source-rate fields feed the fluid every step.  The event-count
    // and energy-loss fields are diagnostic-only and can skip their expensive
    // MPI/normalization passes between diagnostic writes.  Ghost filling stays
    // independently controllable; the production caller currently keeps it on
    // for the three physics fields to preserve the accepted halo semantics.
    void NormalizeAndSync(
        int lev,
        double dt_s,
        bool normalize_diagnostic_fields = true,
        bool fill_boundaries = true);

    amrex::MultiFab& LowElectronSourceRate(int lev);
    amrex::MultiFab& PositiveIonSourceRate(int lev);
    amrex::MultiFab& DirectNegativeIonSourceRate(int lev);
    amrex::MultiFab& IonizationEventCount(int lev);
    amrex::MultiFab& EnergyLossDensityEvPerM3(int lev);

    amrex::MultiFab const& LowElectronSourceRate(int lev) const;
    amrex::MultiFab const& PositiveIonSourceRate(int lev) const;
    amrex::MultiFab const& DirectNegativeIonSourceRate(int lev) const;
    amrex::MultiFab const& IonizationEventCount(int lev) const;
    amrex::MultiFab const& EnergyLossDensityEvPerM3(int lev) const;

    // Rank-local counters for order-1 contributions missing every valid box.
    // ghost = landed in a grow cell and was routed to the owner by the
    // SumBoundary in NormalizeAndSync; clamp-counter = beyond even the grown
    // boxes and deferred for collective routing to the true owning cell.
    std::uint64_t OffrankGhostDepositCount() const noexcept
    {
        return m_offrank_ghost_deposit_count;
    }
    std::uint64_t OffrankClampCount() const noexcept
    {
        return m_offrank_clamp_count;
    }
    void RestoreOffrankDepositCounts(
        std::uint64_t ghost_count,
        std::uint64_t clamp_count) noexcept
    {
        m_offrank_ghost_deposit_count = ghost_count;
        m_offrank_clamp_count = clamp_count;
    }

private:
    struct LocalFabRoute {
        int global_index;
        amrex::Box valid_box;
        amrex::Box grown_box;
    };

    // Deposits landing beyond every local grown box are deferred and routed
    // exactly to the owning cell during NormalizeAndSync; entries are sorted
    // into one deterministic global order first so the per-cell summation
    // order is identical across ranks and layouts.
    struct DeferredOffboxDeposit {
        int field_id;
        int i;
        int j;
        double value;
    };

    void AddToField(int lev, int field_id, double r_m, double z_m, double value);
    void AddToRecipientCell(int lev, int field_id, int i, int j, double value);
    amrex::MultiFab& FieldById(int lev, int field_id);
    void FoldScratchIntoFields(int lev);
    void FlushDeferredOffboxDeposits(int lev);

    amrex::Vector<RreaEventAccumulatorLevelBinding> m_levels;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_s_low_electron;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_s_positive_ion;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_s_direct_negative_ion;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_ionization_event_count;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_energy_loss_density_ev_m3;
    // All event fields share one immutable BoxArray and DistributionMap.
    // Preserve MFIter's local routing order without reconstructing it for
    // every shape-function recipient.
    amrex::Vector<std::vector<LocalFabRoute>> m_local_fab_routes;
    // Deposit scratch, one FAB per local grown box with one component per
    // event field.  Every recipient cell is written here with a plain indexed
    // add instead of a per-recipient MultiFab index lookup and atomic; the
    // whole scratch is folded into the fields once, at NormalizeAndSync, in
    // the order the deposits arrived, so per-cell sums are unchanged.
    amrex::Vector<std::vector<amrex::BaseFab<amrex::Real>>> m_scratch;
    amrex::Vector<std::vector<DeferredOffboxDeposit>> m_deferred_offbox;
    std::uint64_t m_offrank_ghost_deposit_count = 0;
    std::uint64_t m_offrank_clamp_count = 0;
};

}  // namespace rrea

#endif
