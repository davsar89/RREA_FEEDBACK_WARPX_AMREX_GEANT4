// C&D plane-flux stream for RreaWarpXCoupling: the signed crossing tallies the
// transport staged, plus the core-averaged total and ambient Ez at each plane.

#include "RreaWarpXCoupling.H"

#include "RreaCouplingDetail.H"

#include "rrea/RreaFieldGatherStencil.H"
#include "rrea/RreaFieldInitializer.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace rrea::warpx {

using detail::collective_create_dir_all;
using detail::join_path;


void RreaWarpXCoupling::MaybeWriteCdPlaneFlux(
    int step,
    amrex::Real time_s,
    bool force)
{
    if (!m_cd_plane_flux.Due(step, time_s, force)) {
        return;
    }
#if (AMREX_SPACEDIM < 2)
    amrex::Abort("RREA C&D plane diagnostics require an RZ build");
#else
    if (m_level_bindings.size() != 1U
        || m_level_bindings.front().efield_z == nullptr) {
        amrex::Abort(
            "RREA C&D plane diagnostics require one initialized cell-centered field level");
    }
    auto const& binding = m_level_bindings.front();
    auto const& geom = binding.geom;
    auto const& ez = *binding.efield_z;
    auto const& planes = m_cd_plane_flux.PlaneZ();
    std::size_t const plane_count = planes.size();
    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    auto const domain = geom.Domain();
    int const radial_lo = domain.smallEnd(0);
    int const radial_hi = domain.bigEnd(0);
    int const axial_lo = domain.smallEnd(1);
    int const axial_hi = domain.bigEnd(1);

    // One primitive per rule, pinned by the field-gather stencil smoke --
    // do not re-derive the cell-center mapping or the edge clamp inline.
    std::vector<rrea::RreaAxialPlaneInterpolation> interpolation(plane_count);
    for (std::size_t p = 0; p < plane_count; ++p) {
        interpolation[p] = rrea::RreaAxialPlaneInterpolationFor(
            rrea::RreaCellCenteredGatherIndexCoordinate(
                planes[p], plo[1], dx[1], axial_lo),
            axial_lo, axial_hi);
    }

    std::vector<amrex::Real> core_total_sum(plane_count, amrex::Real(0.0));
    std::vector<amrex::Real> core_ambient_sum(plane_count, amrex::Real(0.0));
    std::vector<amrex::Real> core_area(plane_count, amrex::Real(0.0));
    for (amrex::MFIter mfi(ez); mfi.isValid(); ++mfi) {
        amrex::Box const& box = mfi.validbox();
        auto const arr = ez.const_array(mfi);
        int const i_begin = std::max(radial_lo, box.smallEnd(0));
        int const i_end = std::min(radial_hi, box.bigEnd(0));
#if (AMREX_SPACEDIM == 3)
        int const k_begin = box.smallEnd(2);
        int const k_end = box.bigEnd(2);
#else
        // Array4 retains a k index in an RZ/2-D build, but IntVect/Box does
        // not have a third spatial component.  The sole logical plane is k=0.
        int constexpr k_begin = 0;
        int constexpr k_end = 0;
#endif
        for (std::size_t p = 0; p < plane_count; ++p) {
            auto const sample_row = [&](int j, amrex::Real coefficient) {
                if (!(coefficient > amrex::Real(0.0))
                    || j < box.smallEnd(1) || j > box.bigEnd(1)) {
                    return;
                }
                for (int i = i_begin; i <= i_end; ++i) {
                    // pi cancels from every area-weighted mean.
                    amrex::Real const annulus_area = rrea::RreaCoreAnnulusArea(
                        i - radial_lo, plo[0], dx[0],
                        m_cd_plane_core_radius_m);
                    if (!(annulus_area > amrex::Real(0.0))) {
                        continue;
                    }
                    amrex::Real const r_center =
                        plo[0]
                        + (static_cast<amrex::Real>(i - radial_lo)
                              + amrex::Real(0.5))
                            * dx[0];
                    amrex::Real const ambient_ez =
                        RreaFieldInitializer::EvaluateBackgroundEz(
                            r_center, planes[p], m_advance_config.field_initializer);
                    for (int k = k_begin; k <= k_end; ++k) {
                        amrex::Real const weight = coefficient * annulus_area;
                        core_total_sum[p] += weight * arr(i, j, k);
                        core_ambient_sum[p] += weight * ambient_ez;
                        core_area[p] += weight;
                    }
                }
            };
            auto const& location = interpolation[p];
            if (location.lower == location.upper) {
                sample_row(location.lower, amrex::Real(1.0));
            } else {
                sample_row(location.lower, amrex::Real(1.0) - location.upper_weight);
                sample_row(location.upper, location.upper_weight);
            }
        }
    }

    auto global_flux = m_cd_plane_flux.Accumulator().Flatten();
    if (global_flux.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || plane_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        amrex::Abort("RREA C&D plane diagnostic reduction size overflows int");
    }
    amrex::ParallelDescriptor::ReduceRealSum(
        global_flux.data(), static_cast<int>(global_flux.size()));
    amrex::ParallelDescriptor::ReduceRealSum(
        core_total_sum.data(), static_cast<int>(plane_count));
    amrex::ParallelDescriptor::ReduceRealSum(
        core_ambient_sum.data(), static_cast<int>(plane_count));
    amrex::ParallelDescriptor::ReduceRealSum(
        core_area.data(), static_cast<int>(plane_count));

    static std::string const header =
        "format_version,write_index,step,time_s,plane_index,altitude_m,"
        "all_tracked_up_weight,all_tracked_down_weight,"
        "local_runaway_up_weight,local_runaway_down_weight,"
        "e_ge_1mev_up_weight,e_ge_1mev_down_weight,"
        "local_runaway_up_r_weighted_m,local_runaway_down_r_weighted_m,"
        "local_runaway_up_r2_weighted_m2,local_runaway_down_r2_weighted_m2,"
        "core_total_ez_v_per_m,core_ambient_ez_v_per_m,"
        "core_field_perturbation_fraction";
    std::string const path = join_path(m_output_dir, "rrea_cd_plane_flux.csv");
    if (!m_cd_plane_stream_initialized) {
        collective_create_dir_all(m_output_dir);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (m_cd_plane_flux.WriteCount() == 0U && amrex::FileExists(path)
                && std::filesystem::file_size(path) != 0U) {
                amrex::Abort(
                    "fresh RREA C&D plane stream refuses to overwrite a non-empty CSV");
            }
            if (ArchiveCsvIfHeaderChanged(path, header, step)) {
                std::ofstream fresh(path, std::ios::trunc);
                if (!fresh) {
                    amrex::FileOpenFailed(path);
                }
                fresh << header << '\n';
                fresh.flush();
                if (!fresh) {
                    amrex::Abort("RREA failed initializing C&D plane CSV: " + path);
                }
            }
        }
        m_cd_plane_stream_initialized = true;
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream output(path, std::ios::app);
        if (!output) {
            amrex::FileOpenFailed(path);
        }
        for (std::size_t p = 0; p < plane_count; ++p) {
            if (!(core_area[p] > amrex::Real(0.0))) {
                amrex::Abort("RREA C&D plane core field average has zero area");
            }
            amrex::Real const total_ez = core_total_sum[p] / core_area[p];
            amrex::Real const ambient_ez = core_ambient_sum[p] / core_area[p];
            amrex::Real const perturbation = std::abs(total_ez - ambient_ez)
                / std::max(std::abs(ambient_ez), amrex::Real(1.0e-30));
            std::size_t const base = p * RreaPlaneFluxValuesPerPlane;
            output << std::setprecision(17)
                   << 1 << ','
                   << m_cd_plane_flux.WriteCount() << ','
                   << step << ',' << time_s << ',' << p << ','
                   << ProfileAltitudeMslAtZ(planes[p]);
            for (std::size_t value = 0; value < RreaPlaneFluxValuesPerPlane; ++value) {
                output << ',' << global_flux[base + value];
            }
            output << ',' << total_ez
                   << ',' << ambient_ez
                   << ',' << perturbation << '\n';
        }
        output.flush();
        if (!output) {
            amrex::Abort("RREA failed appending C&D plane CSV: " + path);
        }
    }
    m_cd_plane_flux.MarkWritten(step, time_s);
#endif
}

}  // namespace rrea::warpx
