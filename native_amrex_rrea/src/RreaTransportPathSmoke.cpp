#include "rrea/RreaTransportPath.H"
#include "rrea/RreaPlaneFlux.H"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void close(double actual, double expected, double tolerance, char const* label)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(label) + ": expected " + std::to_string(expected)
            + ", got " + std::to_string(actual));
    }
}

void check_piece_partition(
    rrea::RreaTransportPathSegment const& path,
    std::size_t expected_count,
    char const* label)
{
    if (path.in_domain_pieces.size() != expected_count) {
        throw std::runtime_error(
            std::string(label) + ": expected " + std::to_string(expected_count)
            + " pieces, got " + std::to_string(path.in_domain_pieces.size()));
    }
    double fraction = path.in_domain_begin_fraction;
    double length = 0.0;
    for (auto const& piece : path.in_domain_pieces) {
        close(piece.begin_fraction, fraction, 2.0e-14, "piece begin continuity");
        if (!(piece.end_fraction > piece.begin_fraction)) {
            throw std::runtime_error(std::string(label) + ": nonpositive piece");
        }
        double x = 0.0, y = 0.0, z = 0.0;
        path.PointAtPieceFraction(
            piece,
            0.5 * (piece.begin_fraction + piece.end_fraction),
            x,
            y,
            z);
        if (z < path.domain_z_lo_m || z > path.domain_z_hi_m) {
            throw std::runtime_error(std::string(label) + ": piece midpoint is unwrapped");
        }
        fraction = piece.end_fraction;
        length += piece.length_m;
    }
    close(fraction, path.in_domain_end_fraction, 2.0e-14, "piece end coverage");
    close(length, path.in_domain_length_m, 5.0e-14, "piece length sum");
}

}  // namespace

int main()
{
    try {
        {
            std::vector<amrex::Real> const planes{0.0, 1.0, 2.0};
            rrea::RreaPlaneFluxAccumulator flux(planes.size());
            std::vector<std::size_t> order;
            bool const upward_ok = rrea::RreaForEachNonperiodicPlaneCrossing(
                -0.5, 2.0, planes,
                [&](std::size_t index, amrex::Real plane,
                    amrex::Real fraction, int direction) {
                    order.push_back(index);
                    if (direction != 1) {
                        throw std::runtime_error("upward plane direction changed");
                    }
                    close(fraction, (plane + 0.5) / 2.5, 1.0e-14,
                        "upward plane fraction");
                    if (!flux.AddCrossing(
                            index, direction, 3.0,
                            static_cast<amrex::Real>(index + 1U),
                            /*local_runaway=*/index != 0,
                            /*at_least_one_mev=*/index == 2)) {
                        throw std::runtime_error("valid upward plane crossing rejected");
                    }
                });
            if (!upward_ok || order != std::vector<std::size_t>({0, 1, 2})) {
                throw std::runtime_error("upward plane order/coverage changed");
            }

            // Reverse from the exact z=2 endpoint.  The z=2 downward crossing
            // must cancel its upward arrival, while z=1 is crossed downward
            // once and z=0 is not reached.
            order.clear();
            bool const downward_ok = rrea::RreaForEachNonperiodicPlaneCrossing(
                2.0, 0.5, planes,
                [&](std::size_t index, amrex::Real, amrex::Real, int direction) {
                    order.push_back(index);
                    if (!flux.AddCrossing(
                            index, direction, 3.0,
                            static_cast<amrex::Real>(index + 1U),
                            /*local_runaway=*/true,
                            /*at_least_one_mev=*/true)) {
                        throw std::runtime_error("valid downward plane crossing rejected");
                    }
                });
            if (!downward_ok || order != std::vector<std::size_t>({2, 1})) {
                throw std::runtime_error("downward plane order/coverage changed");
            }
            auto const& bins = flux.Bins();
            close(bins[2].selection[0].Net(), 0.0, 0.0,
                "exact-plane recross cancellation");
            close(bins[1].selection[0].Net(), 0.0, 0.0,
                "interior recross cancellation");
            close(bins[0].selection[0].Net(), 3.0, 0.0,
                "uncrossed lower plane net flux");
            close(
                bins[2].local_runaway_radial.upward_first_m
                    - bins[2].local_runaway_radial.downward_first_m,
                0.0,
                0.0,
                "exact-plane signed radial first-moment cancellation");
            close(
                bins[1].local_runaway_radial.upward_second_m2
                    - bins[1].local_runaway_radial.downward_second_m2,
                0.0,
                0.0,
                "interior signed radial second-moment cancellation");
            if (rrea::RreaForEachNonperiodicPlaneCrossing(
                    0.0, 1.0,
                    std::vector<amrex::Real>{0.0, 0.0},
                    [](std::size_t, amrex::Real, amrex::Real, int) {})) {
                throw std::runtime_error("duplicate plane locations were accepted");
            }
            if (flux.AddCrossing(planes.size(), 1, 1.0, 0.0, true, true)
                || flux.AddCrossing(0, 0, 1.0, 0.0, true, true)
                || flux.AddCrossing(0, 1, -1.0, 0.0, true, true)
                || flux.AddCrossing(0, 1, 1.0, -1.0, true, true)) {
                throw std::runtime_error("invalid plane flux input was accepted");
            }
            rrea::RreaPlaneFluxAccumulator overflow_flux(1);
            if (!overflow_flux.AddCrossing(
                    0, 1, std::numeric_limits<amrex::Real>::max(), 0.0,
                    true, true)) {
                throw std::runtime_error("finite maximum plane flux was rejected");
            }
            auto const before_overflow = overflow_flux.Flatten();
            if (overflow_flux.AddCrossing(
                    0, 1, std::numeric_limits<amrex::Real>::max(), 0.0,
                    true, true)
                || overflow_flux.Flatten() != before_overflow) {
                throw std::runtime_error(
                    "overflowing plane flux update was not rejected transactionally");
            }
            rrea::RreaPlaneFluxAccumulator batch_overflow_flux(1);
            amrex::Real const real_max =
                std::numeric_limits<amrex::Real>::max();
            if (!batch_overflow_flux.AddCrossing(
                    0, 1, real_max / amrex::Real(2.0), 0.0,
                    false, false)) {
                throw std::runtime_error(
                    "plane batch-overflow fixture setup failed");
            }
            auto const before_batch_overflow = batch_overflow_flux.Flatten();
            std::vector<rrea::RreaPlaneFluxCrossing> const overflowing_batch{
                {0, 1, real_max / amrex::Real(3.0), 0.0, false, false},
                {0, 1, real_max / amrex::Real(3.0), 0.0, false, false}};
            rrea::RreaPlaneFluxAccumulator::PreparedUpdate rejected_update;
            if (batch_overflow_flux.PrepareCrossings(
                    overflowing_batch, rejected_update)
                || batch_overflow_flux.Flatten() != before_batch_overflow) {
                throw std::runtime_error(
                    "deferred plane batch overflow mutated cumulative state");
            }

            rrea::RreaPlaneFluxAccumulator batch_commit_flux(1);
            std::vector<rrea::RreaPlaneFluxCrossing> const valid_batch{
                {0, 1, 2.0, 3.0, true, false},
                {0, -1, 0.5, 4.0, false, true}};
            rrea::RreaPlaneFluxAccumulator::PreparedUpdate prepared_update;
            auto const before_prepare = batch_commit_flux.Flatten();
            if (!batch_commit_flux.PrepareCrossings(
                    valid_batch, prepared_update)
                || batch_commit_flux.Flatten() != before_prepare) {
                throw std::runtime_error(
                    "valid deferred plane preview mutated live state");
            }
            batch_commit_flux.CommitPrepared(std::move(prepared_update));
            auto const& committed_bin = batch_commit_flux.Bins()[0];
            close(committed_bin.selection[0].upward, 2.0, 0.0,
                "prepared plane upward commit");
            close(committed_bin.selection[0].downward, 0.5, 0.0,
                "prepared plane downward commit");
            close(committed_bin.selection[1].upward, 2.0, 0.0,
                "prepared plane runaway commit");
            close(committed_bin.selection[2].downward, 0.5, 0.0,
                "prepared plane one-MeV commit");
            rrea::RreaPlaneFluxAccumulator moment_flux(1);
            if (!moment_flux.AddCrossing(0, 1, 2.0, 3.0, true, false)) {
                throw std::runtime_error("valid radial plane moment was rejected");
            }
            auto const& moment = moment_flux.Bins()[0].local_runaway_radial;
            close(moment.upward_first_m, 6.0, 0.0,
                "radial first moment accumulation");
            close(moment.upward_second_m2, 18.0, 0.0,
                "radial second moment accumulation");

            auto const flat = flux.Flatten();
            rrea::RreaPlaneFluxAccumulator restored(planes.size());
            if (!restored.RestoreFlat(flat)
                || restored.Bins()[0].selection[0].Net()
                    != bins[0].selection[0].Net()) {
                throw std::runtime_error("plane flux flat checkpoint round trip changed");
            }
            auto invalid_flat = flat;
            invalid_flat[0] = -1.0;
            if (restored.RestoreFlat(invalid_flat)
                || restored.RestoreFlat(std::vector<amrex::Real>{1.0})) {
                throw std::runtime_error("invalid plane flux checkpoint state accepted");
            }

            rrea::RreaPlaneFluxState state;
            state.Configure(planes, 1.0e-8);
            if (state.Due(1, 5.0e-9, false)
                || !state.Due(2, 1.0e-8, false)) {
                throw std::runtime_error("plane flux output schedule due predicate changed");
            }
            state.MarkWritten(2, 1.0e-8);
            if (state.WriteCount() != 1 || state.Due(2, 1.0e-8, true)
                || !state.Due(3, 1.5e-8, true)) {
                throw std::runtime_error("plane flux forced-write/dedup contract changed");
            }

            rrea::RreaPlaneFluxState roundoff_state;
            roundoff_state.Configure(planes, 1.0e-8);
            amrex::Real const one_ulp_early = std::nextafter(
                amrex::Real(1.0e-8), amrex::Real(0.0));
            if (!roundoff_state.Due(8, one_ulp_early, false)) {
                throw std::runtime_error("plane flux cadence lost its roundoff tolerance");
            }
            roundoff_state.MarkWritten(8, one_ulp_early);
            if (roundoff_state.Due(9, amrex::Real(1.125e-8), false)
                || !roundoff_state.Due(16, amrex::Real(2.0e-8), false)) {
                throw std::runtime_error("plane flux cadence duplicated an epsilon-early write");
            }

            rrea::RreaPlaneFluxState forced_state;
            forced_state.Configure(planes, 1.0e-8);
            forced_state.MarkWritten(4, amrex::Real(5.0e-9));
            if (!forced_state.Due(8, amrex::Real(1.0e-8), false)) {
                throw std::runtime_error("forced plane write advanced the regular cadence");
            }
            rrea::RreaPlaneFluxState restored_state;
            restored_state.Configure(planes, 1.0e-8);
            restored_state.RestoreSchedule(
                state.WriteCount(), state.LastWriteStep(),
                state.LastWriteTimeS(), state.NextOutputTimeS());
            if (restored_state.WriteCount() != state.WriteCount()
                || restored_state.LastWriteStep() != state.LastWriteStep()
                || restored_state.NextOutputTimeS() != state.NextOutputTimeS()) {
                throw std::runtime_error("plane flux schedule checkpoint round trip changed");
            }
        }
        {
            auto const path = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.0, 0.5, 0.0, 0.25,
                0.0, 1.0, -1.0, 1.0);
            close(path.full_length_m, 0.25, 1.0e-14, "interior full length");
            close(path.in_domain_length_m, 0.25, 1.0e-14, "interior clipped length");
            if (path.exits_domain) {
                throw std::runtime_error("interior path incorrectly marked escaping");
            }
            check_piece_partition(path, 1, "interior");
        }
        {
            auto const path = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.5, 0.5, 0.0, 1.5,
                0.0, 1.0, -1.0, 1.0);
            close(path.in_domain_end_fraction, 0.5, 1.0e-14, "axial exit fraction");
            close(path.in_domain_length_m, 0.5, 1.0e-14, "axial clipped length");
            if (!path.exits_domain) {
                throw std::runtime_error("axial exit was not detected");
            }
            check_piece_partition(path, 1, "absorbing axial");
        }
        {
            auto const path = rrea::ClipRzTransportPath(
                0.25, 0.0, 0.0, 1.25, 0.0, 0.0,
                0.0, 1.0, -1.0, 1.0);
            close(path.in_domain_end_fraction, 0.75, 1.0e-14, "radial exit fraction");
            close(path.in_domain_length_m, 0.75, 1.0e-14, "radial clipped length");
            if (!path.exits_domain) {
                throw std::runtime_error("radial exit was not detected");
            }
            check_piece_partition(path, 1, "absorbing radial");
        }
        {
            double constexpr pi = 3.141592653589793238462643383279502884;
            auto const path = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.0, 0.5, pi, 0.0,
                0.0, 1.0, -1.0, 1.0);
            close(path.full_length_m, 1.0, 1.0e-14, "Cartesian RZ chord length");
            close(path.in_domain_length_m, 1.0, 1.0e-14, "axis-crossing chord length");
            check_piece_partition(path, 1, "axis crossing");
        }
        {
            auto const path = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.75, 0.5, 0.0, 3.25,
                0.0, 1.0, -1.0, 1.0, true);
            close(path.full_length_m, 2.5, 1.0e-14, "periodic full length");
            close(path.in_domain_length_m, 2.5, 1.0e-14, "periodic material length");
            if (path.exits_domain) {
                throw std::runtime_error("periodic z chord incorrectly marked escaping");
            }
            check_piece_partition(path, 3, "periodic forward multiwrap");
            close(path.in_domain_pieces[0].length_m, 0.25, 1.0e-14, "first seam piece");
            close(path.in_domain_pieces[1].length_m, 2.0, 1.0e-14, "middle seam piece");
            close(path.in_domain_pieces[2].length_m, 0.25, 1.0e-14, "last seam piece");
            double x = 0.0, y = 0.0, z = 0.0;
            path.PointAtFullPathFraction(1.0, x, y, z);
            close(z, -0.75, 1.0e-14, "periodic wrapped endpoint");
        }
        {
            auto const path = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.75, 0.5, 0.0, -5.25,
                0.0, 1.0, -1.0, 1.0, true);
            close(path.full_length_m, 6.0, 1.0e-14, "reverse multiwrap length");
            check_piece_partition(path, 4, "periodic reverse multiwrap");
            close(path.in_domain_pieces[0].length_m, 1.75, 2.0e-14, "reverse first");
            close(path.in_domain_pieces[1].length_m, 2.0, 2.0e-14, "reverse second");
            close(path.in_domain_pieces[2].length_m, 2.0, 2.0e-14, "reverse third");
            close(path.in_domain_pieces[3].length_m, 0.25, 2.0e-14, "reverse last");
            double x = 0.0, y = 0.0, z = 0.0;
            path.PointAtFullPathFraction(1.0, x, y, z);
            close(z, 0.75, 1.0e-14, "reverse wrapped endpoint");
        }
        {
            auto const path = rrea::ClipRzTransportPath(
                0.25, 0.0, 0.75, 1.25, 0.0, 5.75,
                0.0, 1.0, -1.0, 1.0, true);
            close(path.in_domain_end_fraction, 0.75, 1.0e-14, "periodic radial clip");
            if (!path.exits_domain) {
                throw std::runtime_error("periodic radial exit was not detected");
            }
            check_piece_partition(path, 3, "periodic radial clipped multiwrap");
        }
        {
            // A Compton scatter before the incident absorbing-boundary exit gets
            // the unused distance from the complete push, not merely the
            // incident in-domain remainder. An inward scatter can therefore
            // remain in the domain for the rest of the timestep.
            auto const incident_path = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.5, 0.5, 0.0, 1.5,
                0.0, 1.0, 0.0, 1.0);
            close(incident_path.full_length_m, 1.0, 1.0e-14,
                "Compton incident full path");
            close(incident_path.in_domain_length_m, 0.5, 1.0e-14,
                "Compton incident clipped path");
            double const interaction_distance = 0.25;
            double const unused_full_distance =
                incident_path.full_length_m - interaction_distance;
            auto const continuation = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.75,
                0.5, 0.0, 0.75 - unused_full_distance,
                0.0, 1.0, 0.0, 1.0);
            close(continuation.in_domain_length_m, 0.75, 1.0e-14,
                "Compton inward continuation distance");
            if (continuation.exits_domain) {
                throw std::runtime_error(
                    "inward Compton continuation inherited the incident escape");
            }
        }
        {
            // The certified radial domain may be an annulus.  Checking only
            // the continuation endpoint would miss a chord that crosses the
            // absorbing inner cylinder and re-enters before its endpoint.
            double constexpr pi = 3.141592653589793238462643383279502884;
            auto const annular = rrea::ClipRzTransportPath(
                0.75, 0.0, 0.5, 0.75, pi, 0.5,
                0.5, 1.0, 0.0, 1.0);
            if (!annular.exits_domain) {
                throw std::runtime_error(
                    "annular continuation crossed the inner absorber undetected");
            }
            close(annular.in_domain_length_m, 0.25, 1.0e-14,
                "annular inner-boundary distance");
        }
        {
            // Clip fractions alone cannot distinguish an endpoint exactly on
            // an excluded half-open upper face from an interior endpoint.
            // Runtime ownership therefore combines exits_domain with the
            // boundary-mode-aware endpoint predicate.
            auto const exact_axial = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.0, 0.5, 0.0, 1.0,
                0.0, 1.0, -1.0, 1.0);
            if (exact_axial.exits_domain
                || !exact_axial.OutsideCertifiedDomain(0.5, 1.0)) {
                throw std::runtime_error(
                    "exact absorbing axial upper face lost half-open ownership");
            }
            auto const exact_radial = rrea::ClipRzTransportPath(
                0.25, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, -1.0, 1.0);
            if (exact_radial.exits_domain
                || !exact_radial.OutsideCertifiedDomain(1.0, 0.0)) {
                throw std::runtime_error(
                    "exact absorbing radial upper face lost half-open ownership");
            }
        }
        {
            // Ending exactly on a seam must not create a zero-length trailing
            // image.  The endpoint lookup is canonically wrapped to z_lo.
            auto const path = rrea::ClipRzTransportPath(
                0.5, 0.0, 0.0, 0.5, 0.0, 1.0,
                0.0, 1.0, -1.0, 1.0, true);
            check_piece_partition(path, 1, "exact periodic seam endpoint");
            double x = 0.0, y = 0.0, z = 0.0;
            path.PointAtFullPathFraction(1.0, x, y, z);
            close(z, -1.0, 1.0e-14, "exact seam wrapping");
            if (path.OutsideCertifiedDomain(std::hypot(x, y), z)) {
                throw std::runtime_error(
                    "periodic exact upper seam was classified as absorbing");
            }
        }
        std::cout << "rrea_transport_path_smoke passed\n";
    } catch (std::exception const& exc) {
        std::cerr << "rrea_transport_path_smoke failed: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
