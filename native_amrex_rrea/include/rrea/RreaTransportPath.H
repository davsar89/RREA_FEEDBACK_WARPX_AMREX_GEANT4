#ifndef RREA_TRANSPORT_PATH_H
#define RREA_TRANSPORT_PATH_H

#include <AMReX_REAL.H>

#include <vector>

namespace rrea {

// One contiguous portion of a transport chord in a single periodic image of
// the physical z domain.  Fractions always refer to the unwrapped start/end
// chord.  Splitting at every periodic seam makes density, field, and optical-
// depth integration additive even when a particle crosses the axial period
// more than once in one WarpX push.
struct RreaTransportPathPiece {
    amrex::Real begin_fraction = 0.0;
    amrex::Real end_fraction = 0.0;
    amrex::Real length_m = 0.0;
    // Add this shift to the unwrapped z coordinate on the open interior of
    // this piece to obtain the corresponding physical-domain coordinate.
    amrex::Real wrapped_z_shift_m = 0.0;
};

struct RreaTransportPathSegment {
    amrex::Real start_x_m = 0.0;
    amrex::Real start_y_m = 0.0;
    amrex::Real start_z_m = 0.0;
    amrex::Real end_x_m = 0.0;
    amrex::Real end_y_m = 0.0;
    amrex::Real end_z_m = 0.0;
    amrex::Real full_length_m = 0.0;
    amrex::Real in_domain_begin_fraction = 0.0;
    amrex::Real in_domain_end_fraction = 0.0;
    amrex::Real in_domain_length_m = 0.0;
    bool exits_domain = false;
    bool periodic_z = false;
    amrex::Real domain_r_lo_m = 0.0;
    amrex::Real domain_r_hi_m = 0.0;
    amrex::Real domain_z_lo_m = 0.0;
    amrex::Real domain_z_hi_m = 0.0;
    std::vector<RreaTransportPathPiece> in_domain_pieces;

    void PointAtFullPathFraction(
        amrex::Real fraction,
        amrex::Real& x_m,
        amrex::Real& y_m,
        amrex::Real& z_m) const noexcept;
    [[nodiscard]] amrex::Real WrapZ(amrex::Real z_m) const noexcept;
    void PointAtPieceFraction(
        RreaTransportPathPiece const& piece,
        amrex::Real fraction,
        amrex::Real& x_m,
        amrex::Real& y_m,
        amrex::Real& z_m) const noexcept;
    [[nodiscard]] bool OutsideCertifiedDomain(
        amrex::Real r_m, amrex::Real z_m) const noexcept;
};

// Clip a post-push RZ chord against the cylindrical physical domain.  The
// captured start must be in-domain.  Radial exits are clipped; axial exits are
// either clipped (absorbing) or wrapped for density/field lookup (periodic).
// The coupling rejects all other boundary policies before transport.
[[nodiscard]] RreaTransportPathSegment ClipRzTransportPath(
    amrex::Real start_r_m,
    amrex::Real start_theta,
    amrex::Real start_z_m,
    amrex::Real end_r_m,
    amrex::Real end_theta,
    amrex::Real end_z_m,
    amrex::Real domain_r_lo_m,
    amrex::Real domain_r_hi_m,
    amrex::Real domain_z_lo_m,
    amrex::Real domain_z_hi_m,
    bool periodic_z = false);

}  // namespace rrea

#endif
