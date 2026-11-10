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

#include <array>
#include <cstdint>
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

// Material source rates of one step for the fluid, S = sum_k w_k / (V dt),
// for low-energy electrons and positive ions: each event weight w_k is
// deposited with the WarpX order-1 cell-centred shape (cic_axis) and
// normalised by the RZ shell volume V of its cell.
class RreaEventAccumulator {
public:
    RreaEventAccumulator(
        amrex::Geometry const& geom,
        amrex::BoxArray const& box_array,
        amrex::DistributionMapping const& distribution_map,
        int n_grow);

    void Reset();

    void AddLowElectronSource(double r_m, double z_m, double weight)
    {
        AddToField(0, r_m, z_m, weight);
    }
    void AddPositiveIonSource(double r_m, double z_m, double weight)
    {
        AddToField(1, r_m, z_m, weight);
    }

    // Folds the step's deposits into the fields, routes off-box deposits to
    // their owners, divides by the RZ cell volume and dt, and fills ghosts.
    void NormalizeAndSync(double dt_s);

    amrex::MultiFab const& LowElectronSourceRate() const { return m_fields[0]; }
    amrex::MultiFab const& PositiveIonSourceRate() const { return m_fields[1]; }

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

    void AddToField(int field_id, double r_m, double z_m, double value);
    void AddToRecipientCell(int field_id, int i, int j, double value);
    void FoldScratchIntoFields();
    void FlushDeferredOffboxDeposits();

    amrex::Geometry m_geom;
    // Indexed by field id: low-energy electron, positive ion, direct negative
    // ion.  All share one immutable BoxArray and DistributionMap.
    std::array<amrex::MultiFab, 2> m_fields;
    // Preserve MFIter's local routing order without reconstructing it for
    // every shape-function recipient.
    std::vector<LocalFabRoute> m_local_fab_routes;
    std::array<std::size_t, 2> m_last_valid_route{0, 0};
    // Deposit scratch, one FAB per local grown box with one component per
    // event field.  Every recipient cell is written here with a plain indexed
    // add instead of a per-recipient MultiFab index lookup and atomic; the
    // whole scratch is folded into the fields once, at NormalizeAndSync, in
    // the order the deposits arrived, so per-cell sums are unchanged.
    std::vector<amrex::BaseFab<amrex::Real>> m_scratch;
    std::vector<DeferredOffboxDeposit> m_deferred_offbox;
    std::uint64_t m_offrank_ghost_deposit_count = 0;
    std::uint64_t m_offrank_clamp_count = 0;
};

}  // namespace rrea

#endif
