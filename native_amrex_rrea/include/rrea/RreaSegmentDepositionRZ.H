#ifndef RREA_SEGMENT_DEPOSITION_RZ_H
#define RREA_SEGMENT_DEPOSITION_RZ_H

// Charge-conserving deposition of one straight (r,z) segment -- a committed
// net chord or a newborn residual leg -- onto face currents Jr (r-face) and
// Jz (z-face), such that the discrete RZ continuity identity
//
//     rho^{n+1}(c) - rho^{n}(c)
//   = -dt * [ (r_f+ Jr+ - r_f- Jr-) / (r_c dr) + (Jz+ - Jz-) / dz ](c)
//
// holds to round-off at EVERY cell c, where rho is the cell-centered CIC
// (cloud-in-cell) deposit of the particle at its endpoints and the divergence
// carries the same RZ metric weighting used by Maxwell continuity
// (r_f = face radius, r_c = cell-center radius).  m = 0 only: axisymmetric
// charge transport has no theta term in the divergence; J_theta (a ring
// current from v_theta) affects B, never this continuity identity.
//
// Method: Umeda-style zigzag.  The segment is subdivided so no piece crosses
// more than one cell boundary per direction (pieces split at boundary
// crossings), and each in-cell piece deposits its index-space fluxes
// F = q_w * d(index)/dt to the two faces per direction with midpoint CIC
// weights.  Index-space continuity then telescopes exactly; the physical
// r-metric factors cancel between J and the weighted divergence because both
// are defined from the same face/cell radii.  The smoke
// (RreaSegmentDepositionSmoke.cpp) is the arbiter: it asserts the identity
// itself, not this comment.
//
// Radial boundary: a segment may touch r = 0.  CIC support below the axis is
// folded back (|r| reflection), consistent with an even m = 0 density; the
// axis face carries r_f = 0 and therefore no flux, matching the solver's
// exact-zero axis coefficient.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rrea {

struct SegmentGridRZ {
    int nr = 0;             // cells in r
    int nz = 0;             // cells in z
    double dr = 0.0;
    double dz = 0.0;
    double r_lo = 0.0;      // domain low edge in r (usually 0)
    double z_lo = 0.0;
    // Cell-centered rho, size nr*nz, index c = i + nr*j.
    // Jr on r-faces: size (nr+1)*nz, face f = i + (nr+1)*j (i = 0..nr).
    // Jz on z-faces: size nr*(nz+1), face f = i + nr*j     (j = 0..nz).
    std::vector<double> rho;
    std::vector<double> jr;
    std::vector<double> jz;

    void resize()
    {
        rho.assign(static_cast<std::size_t>(nr) * nz, 0.0);
        jr.assign(static_cast<std::size_t>(nr + 1) * nz, 0.0);
        jz.assign(static_cast<std::size_t>(nr) * (nz + 1), 0.0);
    }
    double r_center(int i) const { return r_lo + (i + 0.5) * dr; }
    double r_face(int i) const { return r_lo + i * dr; }
};

// Cell-centered CIC deposit of charge q_w at (r, z); support folded at the
// axis (even reflection) and clamped at the outer/z edges (the smoke keeps
// fixtures interior; production escape handling is the boundary ledger's job).
inline void DepositChargeCIC(SegmentGridRZ& g, double q_w, double r, double z)
{
    double const xr = (r - g.r_lo) / g.dr - 0.5;
    double const xz = (z - g.z_lo) / g.dz - 0.5;
    int const i0 = static_cast<int>(std::floor(xr));
    int const j0 = static_cast<int>(std::floor(xz));
    double const fr = xr - i0;
    double const fz = xz - j0;
    double const wr[2] = {1.0 - fr, fr};
    double const wz[2] = {1.0 - fz, fz};
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            int i = i0 + a;
            int j = j0 + b;
            if (i < 0) { i = -1 - i; }              // axis fold (even m=0)
            if (i >= g.nr) { i = g.nr - 1; }
            if (j < 0) { j = 0; }
            if (j >= g.nz) { j = g.nz - 1; }
            g.rho[static_cast<std::size_t>(i) + static_cast<std::size_t>(g.nr) * j]
                += q_w * wr[a] * wz[b];
        }
    }
}

namespace detail {

// One in-cell piece: deposit index-space fluxes with midpoint CIC weights
// (Umeda 2003, second-order equivalent to Esirkepov order 1 for CIC rho).
inline void deposit_piece(
    SegmentGridRZ& g, double q_w_over_dt,
    double xr0, double xz0, double xr1, double xz1)
{
    // Index-space displacement and midpoint (in the CIC frame: cell-center
    // coordinates, i.e. xr = (r - r_lo)/dr - 0.5).
    double fr_flux = q_w_over_dt * (xr1 - xr0);
    double const fz_flux = q_w_over_dt * (xz1 - xz0);
    double const mr = 0.5 * (xr0 + xr1);
    double const mz = 0.5 * (xz0 + xz1);
    // The r-flux crosses the r-face between floor cells; weights along z.
    {
        int const jf = static_cast<int>(std::floor(mz));
        double const fz = mz - jf;
        // For a piece confined to node-interval (k, k+1) the flux face is
        // ALWAYS k+1/2 (array index k+1): Delta(Wr) has support {k, k+1}
        // only, so one face carries the whole 1D flux. The midpoint enters
        // the TRANSVERSE weights only (exact: D(WxWy)=DWx*avgWy+avgWx*DWy).
        int const fi = static_cast<int>(std::floor(mr)) + 1;
        // Axis rule (derived against folded CIC, verified by the smoke): a
        // piece whose interval sits below node 0 (fi <= 0, only reachable
        // via the sub-half-cell axis region) moves charge WITHIN the folded
        // node-0 cloud -- folded rho does not change, so its radial flux is
        // dropped, matching r_f = 0 at the axis face.
        if (fi <= 0) { fr_flux = 0.0; }
        double const wz0 = 1.0 - fz;
        for (int b = 0; b < 2; ++b) {
            int j = jf + b;
            if (j < 0) { j = 0; }
            if (j >= g.nz) { j = g.nz - 1; }
            int i = fi;
            if (i <= 0) { continue; }
            if (i > g.nr) { i = g.nr; }
            g.jr[static_cast<std::size_t>(i)
                 + static_cast<std::size_t>(g.nr + 1) * j]
                += fr_flux * (b == 0 ? wz0 : fz);
        }
    }
    // The z-flux; weights along r (with axis fold).
    {
        int const ifc = static_cast<int>(std::floor(mr));
        double const fr = mr - ifc;
        int const fj = static_cast<int>(std::floor(mz)) + 1;
        double const wr0 = 1.0 - fr;
        for (int a = 0; a < 2; ++a) {
            int i = ifc + a;
            if (i < 0) { i = -1 - i; }
            if (i >= g.nr) { i = g.nr - 1; }
            int j = fj;
            if (j < 0) { j = 0; }
            if (j > g.nz) { j = g.nz; }
            g.jz[static_cast<std::size_t>(i)
                 + static_cast<std::size_t>(g.nr) * j]
                += fz_flux * (a == 0 ? wr0 : fr);
        }
    }
}

} // namespace detail

// Deposit the segment (r0,z0) -> (r1,z1) flown in dt by charge q_w.
// Splits at cell-boundary crossings in the CIC index frame so each piece
// stays within one CIC cell per direction.
inline void DepositSegmentRZ(
    SegmentGridRZ& g, double q_w, double dt,
    double r0, double z0, double r1, double z1)
{
    double const q_w_over_dt = q_w / dt;
    double xr0 = (r0 - g.r_lo) / g.dr - 0.5;
    double xz0 = (z0 - g.z_lo) / g.dz - 0.5;
    double const xr1 = (r1 - g.r_lo) / g.dr - 0.5;
    double const xz1 = (z1 - g.z_lo) / g.dz - 0.5;
    // Collect crossing fractions of CIC cell boundaries (integers in the
    // index frame) along the segment, both directions, then walk pieces.
    std::vector<double> cuts = {0.0, 1.0};
    auto add_cuts = [&cuts](double a, double b) {
        if (a == b) { return; }
        double const lo = std::min(a, b);
        double const hi = std::max(a, b);
        for (double k = std::ceil(lo); k <= std::floor(hi); ++k) {
            double const s = (k - a) / (b - a);
            if (s > 0.0 && s < 1.0) { cuts.push_back(s); }
        }
    };
    add_cuts(xr0, xr1);
    add_cuts(xz0, xz1);
    std::sort(cuts.begin(), cuts.end());
    for (std::size_t p = 0; p + 1 < cuts.size(); ++p) {
        double const s0 = cuts[p];
        double const s1 = cuts[p + 1];
        if (s1 <= s0) { continue; }
        detail::deposit_piece(
            g, q_w_over_dt,
            xr0 + s0 * (xr1 - xr0), xz0 + s0 * (xz1 - xz0),
            xr0 + s1 * (xr1 - xr0), xz0 + s1 * (xz1 - xz0));
    }
}

// Absorbing faces supported by the production RZ boundary contract.  The
// symmetry axis is not an absorber.  More than one bit is allowed for an
// exact corner crossing.
inline constexpr std::uint32_t kEscapeFaceRHi = 1U;
inline constexpr std::uint32_t kEscapeFaceZLo = 2U;
inline constexpr std::uint32_t kEscapeFaceZHi = 4U;

struct EscapingSegmentDepositResult {
    bool valid = false;
    // Signed charge carried by the explicit final boundary-cloud closure.
    // The earlier part of the escaping chord generally carries the rest of
    // the macro charge through the physical face.
    double explicit_boundary_charge = 0.0;
    // Signed q*dz contribution of the explicit low/high-z face sink.  This is
    // the additional axial current moment beyond the geometric chord.
    double explicit_z_displacement_charge = 0.0;
};

// Deposit one complete charged escape atomically:
//
//   trajectory_start -> final_microchord_start -> exact_boundary_crossing
//
// followed by outward face flux that removes the portion of the order-1 CIC
// cloud at the crossing which is still supported by valid physical cells.
// Merely ending DepositSegmentRZ on the face leaves that portion behind.  For
// an exact planar crossing it is one half of the cloud; at a two-face corner
// it is one quarter.  Corner residual charge is shared equally by the active
// outward faces.  The split is immaterial to charge conservation but avoids a
// coordinate-order preference in the emitted current.
//
// `physical_z_begin` is the first physical z-cell in the padded SegmentGridRZ
// (production uses two low-z pad cells).  rho is updated on both sides of each
// physical boundary face as a scratch-grid transfer.  Consequently the full
// padded identity remains exact while, on the physical cells, rho is exactly
// the deletion of the starting particle cloud.
inline EscapingSegmentDepositResult DepositEscapingSegmentRZ(
    SegmentGridRZ& g,
    double q_w,
    double dt,
    double start_r,
    double start_z,
    double final_microchord_start_r,
    double final_microchord_start_z,
    double crossing_r,
    double crossing_z,
    int physical_nr,
    int physical_z_begin,
    int physical_nz,
    std::uint32_t faces)
{
    EscapingSegmentDepositResult result;
    std::uint32_t constexpr known_faces =
        kEscapeFaceRHi | kEscapeFaceZLo | kEscapeFaceZHi;
    if (!(dt > 0.0) || !std::isfinite(q_w) || !std::isfinite(dt)
        || !std::isfinite(start_r) || !std::isfinite(start_z)
        || !std::isfinite(final_microchord_start_r)
        || !std::isfinite(final_microchord_start_z)
        || !std::isfinite(crossing_r) || !std::isfinite(crossing_z)
        || physical_nr <= 0 || physical_nr >= g.nr
        || physical_nz <= 0 || physical_z_begin <= 0
        || physical_z_begin + physical_nz >= g.nz
        || faces == 0U || (faces & ~known_faces) != 0U
        || ((faces & kEscapeFaceZLo) != 0U
            && (faces & kEscapeFaceZHi) != 0U)) {
        return result;
    }

    // The geometry clip owns which face was crossed.  Snap to that exact face
    // before forming CIC weights so a few ulps of quadratic-root roundoff can
    // never turn a half cloud into 0.50000000000000x.
    if ((faces & kEscapeFaceRHi) != 0U) {
        crossing_r = g.r_lo + physical_nr * g.dr;
    }
    if ((faces & kEscapeFaceZLo) != 0U) {
        crossing_z = g.z_lo + physical_z_begin * g.dz;
    } else if ((faces & kEscapeFaceZHi) != 0U) {
        crossing_z = g.z_lo
            + (physical_z_begin + physical_nz) * g.dz;
    }

    struct BoundaryMove {
        std::size_t cell_from = 0;
        std::size_t cell_to = 0;
        std::size_t face_index = 0;
        double charge = 0.0;
        bool radial = false;
        int orientation = 1;  // coordinate current: +r/+z or -z
    };
    std::vector<BoundaryMove> moves;
    moves.reserve(8);

    double const xr = (crossing_r - g.r_lo) / g.dr - 0.5;
    double const xz = (crossing_z - g.z_lo) / g.dz - 0.5;
    int const i0 = static_cast<int>(std::floor(xr));
    int const j0 = static_cast<int>(std::floor(xz));
    double const fr = xr - i0;
    double const fz = xz - j0;
    double const wr[2] = {1.0 - fr, fr};
    double const wz[2] = {1.0 - fz, fz};
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            int i = i0 + a;
            int j = j0 + b;
            if (i < 0) { i = -1 - i; }
            if (i < 0 || i >= g.nr || j < 0 || j >= g.nz) {
                return result;
            }
            bool const in_physical = i < physical_nr
                && j >= physical_z_begin
                && j < physical_z_begin + physical_nz;
            if (!in_physical) {
                continue;
            }
            std::array<std::uint32_t, 3> candidate_faces{};
            int candidate_count = 0;
            if ((faces & kEscapeFaceRHi) != 0U
                && i == physical_nr - 1) {
                candidate_faces[candidate_count++] = kEscapeFaceRHi;
            }
            if ((faces & kEscapeFaceZLo) != 0U
                && j == physical_z_begin) {
                candidate_faces[candidate_count++] = kEscapeFaceZLo;
            }
            if ((faces & kEscapeFaceZHi) != 0U
                && j == physical_z_begin + physical_nz - 1) {
                candidate_faces[candidate_count++] = kEscapeFaceZHi;
            }
            if (candidate_count == 0) {
                return result;
            }
            double const cell_charge = q_w * wr[a] * wz[b];
            double const share = cell_charge / candidate_count;
            std::size_t const from = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(g.nr) * j;
            for (int candidate = 0; candidate < candidate_count; ++candidate) {
                std::uint32_t const face = candidate_faces[candidate];
                if (face == kEscapeFaceRHi) {
                    moves.push_back(BoundaryMove{
                        from,
                        static_cast<std::size_t>(i + 1)
                            + static_cast<std::size_t>(g.nr) * j,
                        static_cast<std::size_t>(physical_nr)
                            + static_cast<std::size_t>(g.nr + 1) * j,
                        share,
                        true,
                        1});
                } else {
                    int const face_j = face == kEscapeFaceZLo
                        ? physical_z_begin
                        : physical_z_begin + physical_nz;
                    int const to_j = face == kEscapeFaceZLo ? j - 1 : j + 1;
                    moves.push_back(BoundaryMove{
                        from,
                        static_cast<std::size_t>(i)
                            + static_cast<std::size_t>(g.nr) * to_j,
                        static_cast<std::size_t>(i)
                            + static_cast<std::size_t>(g.nr) * face_j,
                        share,
                        false,
                        face == kEscapeFaceZLo ? -1 : 1});
                }
            }
            result.explicit_boundary_charge += cell_charge;
        }
    }
    if (moves.empty()) {
        return result;
    }

    DepositChargeCIC(g, -q_w, start_r, start_z);
    DepositChargeCIC(g, q_w, crossing_r, crossing_z);
    DepositSegmentRZ(
        g, q_w, dt,
        start_r, start_z,
        final_microchord_start_r, final_microchord_start_z);
    DepositSegmentRZ(
        g, q_w, dt,
        final_microchord_start_r, final_microchord_start_z,
        crossing_r, crossing_z);
    for (auto const& move : moves) {
        double const coordinate_flux = move.orientation * move.charge / dt;
        if (move.radial) {
            g.jr[move.face_index] += coordinate_flux;
        } else {
            g.jz[move.face_index] += coordinate_flux;
            result.explicit_z_displacement_charge +=
                move.orientation * move.charge * g.dz;
        }
        // Scratch representation of charge leaving the physical domain.  The
        // destination is a pad cell, never a physical Maxwell cell.
        g.rho[move.cell_from] -= move.charge;
        g.rho[move.cell_to] += move.charge;
    }
    result.valid = true;
    return result;
}

// Convert the padded index-space fluxes produced above to the physical
// current densities consumed by the Maxwell faces.  Only current whose face
// and transverse order-1 support belong to the physical mesh is retained.
// Absorbing-boundary CIC support in guard/pad cells is lost, exactly as it is
// in WarpX DepositCharge (and in RreaEventAccumulator's pinned stencil);
// clamping that current back onto the edge cell violates cellwise continuity.
// A worldline that exits still carries its normal current on the physical
// boundary face, including the explicit residual-cloud flux above.
inline void ConvertPaddedIndexCurrentToPhysicalRZ(
    int nr, int nz, double dr, double dz, int pad_cells,
    double charge_per_index_flux_c,
    std::vector<double> const& jr_index,
    std::vector<double> const& jz_index,
    std::vector<double>& jr_a_per_m2,
    std::vector<double>& jz_a_per_m2)
{
    double constexpr pi = 3.14159265358979323846;
    int const nr_pad = nr + pad_cells;
    jr_a_per_m2.assign(
        static_cast<std::size_t>(nr + 1) * nz, 0.0);
    jz_a_per_m2.assign(
        static_cast<std::size_t>(nr) * (nz + 1), 0.0);

    for (int j = pad_cells; j < pad_cells + nz; ++j) {
        int const j_phys = j - pad_cells;
        for (int i = 0; i <= nr; ++i) {
            int const i_phys = i;
            double const flux = jr_index[
                static_cast<std::size_t>(i)
                + static_cast<std::size_t>(nr_pad + 1) * j];
            double const r_face = i_phys * dr;
            if (flux == 0.0 || r_face <= 0.0) {
                continue;  // the axis face has zero area and carries no flux
            }
            jr_a_per_m2[
                static_cast<std::size_t>(i_phys)
                + static_cast<std::size_t>(nr + 1) * j_phys]
                += charge_per_index_flux_c * flux
                    / (2.0 * pi * r_face * dz);
        }
    }

    for (int j = pad_cells; j <= pad_cells + nz; ++j) {
        int const j_phys = j - pad_cells;
        for (int i = 0; i < nr; ++i) {
            int const i_phys = i;
            double const flux = jz_index[
                static_cast<std::size_t>(i)
                + static_cast<std::size_t>(nr_pad) * j];
            if (flux == 0.0) {
                continue;
            }
            double const r_lo_face = i_phys * dr;
            double const r_hi_face = (i_phys + 1) * dr;
            double const ring_area = pi
                * (r_hi_face * r_hi_face - r_lo_face * r_lo_face);
            jz_a_per_m2[
                static_cast<std::size_t>(i_phys)
                + static_cast<std::size_t>(nr) * j_phys]
                += charge_per_index_flux_c * flux / ring_area;
        }
    }
}

// Index-space divergence the deposit identity is asserted against.  The RZ
// metric factors cancel between J and the divergence when both are built
// from the same face/cell radii, so the plain difference is what telescopes;
// ConvertPaddedIndexCurrentToPhysicalRZ owns the A/m^2 conversion and
// MaxwellTMRZ::DivergenceRZ the physical-space operator.
inline double WeightedDivergence(
    SegmentGridRZ const& g, int i, int j)
{
    std::size_t const jr_lo = static_cast<std::size_t>(i)
        + static_cast<std::size_t>(g.nr + 1) * j;
    std::size_t const jz_lo = static_cast<std::size_t>(i)
        + static_cast<std::size_t>(g.nr) * j;
    return (g.jr[jr_lo + 1] - g.jr[jr_lo])
        + (g.jz[jz_lo + g.nr] - g.jz[jz_lo]);
}

} // namespace rrea

#endif // RREA_SEGMENT_DEPOSITION_RZ_H
