// Video-frame capture for RreaWarpXCoupling: the frame schedule, the per-frame
// species/ion/field grids and moments, and the video-v4 stream writer.

#include "RreaWarpXCoupling.H"

#include "RreaCheckpointJson.H"
#include "RreaCouplingDetail.H"
#include "RreaParticleSweep.H"
#include "RreaVideoFrameCodec.H"

#include "rrea/RreaCsvTextUtil.H"
#include "rrea/RreaReducedObservables.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rrea::warpx {

using detail::collective_create_dir_all;
using detail::global_count_sum;
using detail::global_real_min;
using detail::global_vector_max;
using detail::global_vector_sum;
using detail::join_path;

namespace {

using rrea::cell_index_origin;
using rrea::cell_volume_rz;
using rrea::csv_text::split_csv_line;

// Index of a named schedule column, or the documented positional fallback.
int column_index(
    std::vector<std::string> const& columns,
    std::string const& name,
    int fallback)
{
    for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
        if (columns[i] == name) {
            return i;
        }
    }
    return fallback;
}


struct VideoParticleMoments {
    std::uint64_t macro_count = 0;
    amrex::Real weight = amrex::Real(0.0);
    amrex::Real r_sum = amrex::Real(0.0);
    amrex::Real r2_sum = amrex::Real(0.0);
    amrex::Real z_sum = amrex::Real(0.0);
    amrex::Real z2_sum = amrex::Real(0.0);
    amrex::Real kinetic_sum_eV = amrex::Real(0.0);
    amrex::Real uz_over_u_sum = amrex::Real(0.0);
    amrex::Real radial_over_u_sum = amrex::Real(0.0);
    amrex::Real abs_ur_over_abs_uz_sum = amrex::Real(0.0);
    amrex::Real positive_uz_weight = amrex::Real(0.0);
    amrex::Real negative_uz_weight = amrex::Real(0.0);
    amrex::Real min_r = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real max_r = amrex::Real(0.0);
    amrex::Real min_z = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real max_z = -std::numeric_limits<amrex::Real>::infinity();
};


struct VideoFieldDirectionMoments {
    amrex::Real volume = amrex::Real(0.0);
    amrex::Real abs_er_volume_sum = amrex::Real(0.0);
    amrex::Real abs_ez_volume_sum = amrex::Real(0.0);
    amrex::Real abs_e_volume_sum = amrex::Real(0.0);
    amrex::Real er_over_ez_volume_sum = amrex::Real(0.0);
    amrex::Real max_abs_er = amrex::Real(0.0);
    amrex::Real max_abs_ez = amrex::Real(0.0);
    amrex::Real max_abs_e = amrex::Real(0.0);
    amrex::Real max_er_over_ez = amrex::Real(0.0);
};


// Volume of video bin (ir, *): the shell between the bin's radial edges.
amrex::Real video_bin_volume_rz(
    int ir,
    int nr,
    int nz,
    amrex::Real r_min_m,
    amrex::Real r_max_m,
    amrex::Real z_min_m,
    amrex::Real z_max_m)
{
    amrex::Real const dr = (r_max_m - r_min_m) / static_cast<amrex::Real>(nr);
    amrex::Real const r_inner = r_min_m + static_cast<amrex::Real>(ir) * dr;
    return static_cast<amrex::Real>(rz_shell_volume(
        r_inner,
        r_inner + dr,
        (z_max_m - z_min_m) / static_cast<amrex::Real>(nz)));
}

// Volume shared by a mesh cell and a video bin, from their already-intersected
// radial and axial extents; a degenerate intersection has zero volume.
amrex::Real rz_overlap_volume(
    amrex::Real r_inner,
    amrex::Real r_outer,
    amrex::Real z0,
    amrex::Real z1)
{
    return r_outer <= r_inner || z1 <= z0
        ? amrex::Real(0.0)
        : static_cast<amrex::Real>(rz_shell_volume(r_inner, r_outer, z1 - z0));
}

}  // namespace

void RreaWarpXCoupling::LoadVideoFrameSchedule()
{
    if (m_video_schedule_loaded) {
        return;
    }
    if (!m_video_diag_enable) {
        m_video_schedule_loaded = true;
        return;
    }
    if (!amrex::FileExists(m_video_diag_frame_schedule_path)) {
        amrex::Abort(
            "RREA video frame schedule file not found: "
            + m_video_diag_frame_schedule_path);
    }

    std::ifstream input(m_video_diag_frame_schedule_path);
    if (!input) {
        amrex::FileOpenFailed(m_video_diag_frame_schedule_path);
    }
    std::string header;
    if (!std::getline(input, header)) {
        amrex::Abort("RREA video frame schedule is empty: " + m_video_diag_frame_schedule_path);
    }
    auto const columns = split_csv_line(header);
    int const capture_index_col = column_index(columns, "capture_index", 0);
    int const step_col = column_index(columns, "step", 1);
    int const time_col = column_index(columns, "target_time_s", 2);
    if (step_col < 0 || step_col >= static_cast<int>(columns.size())) {
        amrex::Abort("RREA video frame schedule requires a step column");
    }

    m_video_frames.clear();
    std::string line;
    int row = 1;
    while (std::getline(input, line)) {
        ++row;
        if (line.empty()) {
            continue;
        }
        auto const fields = split_csv_line(line);
        if (step_col >= static_cast<int>(fields.size())) {
            amrex::Abort(
                "RREA video frame schedule row has too few columns: row "
                + std::to_string(row));
        }
        RreaVideoFrameRequest request;
        request.capture_index =
            capture_index_col < static_cast<int>(fields.size())
            ? std::stoi(fields[capture_index_col])
            : static_cast<int>(m_video_frames.size());
        request.step = std::stoi(fields[step_col]);
        request.target_time_s =
            time_col < static_cast<int>(fields.size())
            ? static_cast<amrex::Real>(std::stod(fields[time_col]))
            : amrex::Real(0.0);
        if (request.step < 0) {
            amrex::Abort("RREA video frame schedule contains a negative step");
        }
        m_video_frames.push_back(request);
    }
    std::sort(
        m_video_frames.begin(),
        m_video_frames.end(),
        [](RreaVideoFrameRequest const& lhs, RreaVideoFrameRequest const& rhs) {
            if (lhs.step != rhs.step) {
                return lhs.step < rhs.step;
            }
            return lhs.capture_index < rhs.capture_index;
        });
    m_next_video_frame = 0;
    m_video_schedule_loaded = true;
    amrex::Print() << "RREA video diagnostic loaded "
                   << m_video_frames.size()
                   << " frame requests from "
                   << m_video_diag_frame_schedule_path
                   << "\n";
}

void RreaWarpXCoupling::MaybeWriteVideoFrame(
    WarpX& warpx,
    int step,
    amrex::Real time_s)
{
    if (!m_video_diag_enable || !m_initialized || m_level_bindings.empty()) {
        return;
    }
    LoadVideoFrameSchedule();
    while (m_next_video_frame < m_video_frames.size()
           && m_video_frames[m_next_video_frame].step < step) {
        ++m_next_video_frame;
    }
    if (m_next_video_frame >= m_video_frames.size()
        || m_video_frames[m_next_video_frame].step != step) {
        return;
    }

    auto const& binding = m_level_bindings[0];
    auto const prob_lo = binding.geom.ProbLoArray();
    auto const prob_hi = binding.geom.ProbHiArray();
    amrex::Real const domain_r_min_m = prob_lo[0];
    amrex::Real const domain_r_max_m = prob_hi[0];
#if (AMREX_SPACEDIM >= 2)
    amrex::Real const domain_z_min_m = prob_lo[1];
    amrex::Real const domain_z_max_m = prob_hi[1];
#else
    amrex::Real const domain_z_min_m = amrex::Real(0.0);
    amrex::Real const domain_z_max_m = amrex::Real(1.0);
#endif
    amrex::Real const r_min_m =
        std::isfinite(static_cast<double>(m_video_diag_r_min_m))
        ? amrex::max(domain_r_min_m, m_video_diag_r_min_m)
        : domain_r_min_m;
    amrex::Real const r_max_m =
        std::isfinite(static_cast<double>(m_video_diag_r_max_m))
        ? amrex::min(domain_r_max_m, m_video_diag_r_max_m)
        : domain_r_max_m;
    amrex::Real const z_min_m =
        std::isfinite(static_cast<double>(m_video_diag_z_min_m))
        ? amrex::max(domain_z_min_m, m_video_diag_z_min_m)
        : domain_z_min_m;
    amrex::Real const z_max_m =
        std::isfinite(static_cast<double>(m_video_diag_z_max_m))
        ? amrex::min(domain_z_max_m, m_video_diag_z_max_m)
        : domain_z_max_m;
    if (r_max_m <= r_min_m || z_max_m <= z_min_m) {
        amrex::Abort("RREA video diagnostic requires positive RZ extents");
    }

    int const nr = m_video_diag_nr;
    int const nz = m_video_diag_nz;
    std::size_t const cell_count =
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz);
    std::vector<amrex::Real> electron_weight(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> positive_ion_density_volume_sum(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> positive_ion_volume(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> field_volume_sum(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> field_volume(cell_count, amrex::Real(0.0));
    // Photon and positron
    // population grids are captured per frame alongside the electron map so
    // a species video (e-/e+/gamma) can be rendered later.  The same video
    // energy threshold applies to the photon energy and the positron kinetic
    // energy; disabled species leave all-zero arrays so the frame layout is
    // uniform.
    std::vector<amrex::Real> photon_weight_grid(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> positron_weight_grid(cell_count, amrex::Real(0.0));
    VideoParticleMoments particle_moments;
    VideoFieldDirectionMoments field_moments;
    amrex::Real const video_dr = (r_max_m - r_min_m) / static_cast<amrex::Real>(nr);
    amrex::Real const video_dz = (z_max_m - z_min_m) / static_cast<amrex::Real>(nz);
    amrex::Real const active_z_min =
        std::isfinite(static_cast<double>(m_profile_active_z_min_m))
        ? m_profile_active_z_min_m
        : z_min_m;
    amrex::Real const active_z_max =
        std::isfinite(static_cast<double>(m_profile_active_z_max_m))
        ? m_profile_active_z_max_m
        : z_max_m;

    // Frame bin of a position inside the [r_min, r_max) x [z_min, z_max)
    // window; the caller has already rejected positions outside it.
    auto frame_bin = [&](amrex::Real r, amrex::Real z) {
        int const ir = std::min(nr - 1, std::max(0, static_cast<int>(
            std::floor((r - r_min_m) / (r_max_m - r_min_m)
                       * static_cast<amrex::Real>(nr)))));
        int const iz = std::min(nz - 1, std::max(0, static_cast<int>(
            std::floor((z - z_min_m) / (z_max_m - z_min_m)
                       * static_cast<amrex::Real>(nz)))));
        return static_cast<std::size_t>(iz) * nr + ir;
    };
    // Population grid of one species, binned by macro weight above the frame's
    // energy threshold.  A photon's energy in eV is |u|; a charged macro's is
    // its kinetic energy.
    auto bin_species_weight = [&](
        std::string const& species_name,
        bool massless,
        std::vector<amrex::Real>& out_grid) {
        RreaForEachValidParticle(
            warpx.GetPartContainer().GetParticleContainerFromName(species_name),
            [&](RreaLiveParticle const& p) {
                if (p.r < r_min_m || p.r >= r_max_m
                    || p.z < z_min_m || p.z >= z_max_m
                    || (massless ? p.PhotonEnergyEv() : p.KineticEnergyEv())
                        < m_video_diag_energy_threshold_eV) {
                    return;
                }
                out_grid[frame_bin(p.r, p.z)] += p.weight;
            });
    };

    RreaForEachValidParticle(
        warpx.GetPartContainer().GetParticleContainerFromName(m_seed_species_name),
        [&](RreaLiveParticle const& p) {
            if (p.r < r_min_m || p.r >= r_max_m
                || p.z < z_min_m || p.z >= z_max_m) {
                return;
            }
            amrex::Real const kinetic_eV = p.KineticEnergyEv();
            if (kinetic_eV < m_video_diag_energy_threshold_eV) {
                return;
            }
            amrex::Real const particle_weight = p.weight;
            amrex::Real const u_norm =
                std::sqrt(p.ux * p.ux + p.uy * p.uy + p.uz * p.uz);
            amrex::Real const dir_z =
                u_norm > amrex::Real(0.0) ? p.uz / u_norm : amrex::Real(1.0);
            amrex::Real const radial_dir = u_norm > amrex::Real(0.0)
                ? (p.ux * std::cos(p.theta) + p.uy * std::sin(p.theta)) / u_norm
                : amrex::Real(0.0);
            ++particle_moments.macro_count;
            particle_moments.weight += particle_weight;
            particle_moments.r_sum += particle_weight * p.r;
            particle_moments.r2_sum += particle_weight * p.r * p.r;
            particle_moments.z_sum += particle_weight * p.z;
            particle_moments.z2_sum += particle_weight * p.z * p.z;
            particle_moments.kinetic_sum_eV += particle_weight * kinetic_eV;
            particle_moments.uz_over_u_sum += particle_weight * dir_z;
            particle_moments.radial_over_u_sum += particle_weight * radial_dir;
            particle_moments.abs_ur_over_abs_uz_sum +=
                particle_weight * std::abs(radial_dir)
                / amrex::max(std::abs(dir_z), amrex::Real(1.0e-12));
            if (dir_z > amrex::Real(0.0)) {
                particle_moments.positive_uz_weight += particle_weight;
            } else if (dir_z < amrex::Real(0.0)) {
                particle_moments.negative_uz_weight += particle_weight;
            }
            particle_moments.min_r = amrex::min(particle_moments.min_r, p.r);
            particle_moments.max_r = amrex::max(particle_moments.max_r, p.r);
            particle_moments.min_z = amrex::min(particle_moments.min_z, p.z);
            particle_moments.max_z = amrex::max(particle_moments.max_z, p.z);
            electron_weight[frame_bin(p.r, p.z)] += particle_weight;
        });
    bin_species_weight(m_photon_species_name, /*massless=*/true,
                       photon_weight_grid);
    bin_species_weight(m_positron_species_name, /*massless=*/false,
                       positron_weight_grid);

    auto const dx = binding.geom.CellSizeArray();
    auto const plo = cell_index_origin(binding.geom);
    auto const cell_center = [&](int i, int j) {
#if (AMREX_SPACEDIM >= 2)
        return std::make_pair(
            plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0],
            plo[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1]);
#else
        amrex::ignore_unused(j);
        return std::make_pair(
            plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0],
            amrex::Real(0.0));
#endif
    };
    // Area-exact rebin of one mesh cell onto the frame grid: every video bin
    // the cell overlaps receives value * V_overlap plus the overlap volume, so
    // the frame reports a volume-weighted mean.  Cell volume scales as
    // 2 pi r dr dz, so accumulating one count per cell instead would
    // over-weight small-radius cells and bias the frame toward the axis.
    auto const rebin_cell = [&](int i, int j, amrex::Real value,
                                std::vector<amrex::Real>& weighted,
                                std::vector<amrex::Real>& volume) {
        auto const [r, z] = cell_center(i, j);
        amrex::Real const cell_r_lo = r - amrex::Real(0.5) * dx[0];
        amrex::Real const cell_r_hi = r + amrex::Real(0.5) * dx[0];
#if (AMREX_SPACEDIM >= 2)
        amrex::Real const cell_z_lo = z - amrex::Real(0.5) * dx[1];
        amrex::Real const cell_z_hi = z + amrex::Real(0.5) * dx[1];
#else
        amrex::Real const cell_z_lo = z_min_m;
        amrex::Real const cell_z_hi = z_max_m;
#endif
        if (cell_r_hi <= r_min_m || cell_r_lo >= r_max_m
            || cell_z_hi <= z_min_m || cell_z_lo >= z_max_m) {
            return;
        }
        auto const bin_span = [](amrex::Real lo, amrex::Real hi,
                                 amrex::Real window_lo, amrex::Real window_hi,
                                 amrex::Real bin_width, int bins) {
            auto const clamp = [bins](double value) {
                return std::min(bins - 1, std::max(0, static_cast<int>(value)));
            };
            return std::make_pair(
                clamp(std::floor(
                    (amrex::max(lo, window_lo) - window_lo) / bin_width)),
                clamp(std::ceil(
                    (amrex::min(hi, window_hi) - window_lo) / bin_width) - 1));
        };
        auto const [ir_lo, ir_hi] =
            bin_span(cell_r_lo, cell_r_hi, r_min_m, r_max_m, video_dr, nr);
        auto const [iz_lo, iz_hi] =
            bin_span(cell_z_lo, cell_z_hi, z_min_m, z_max_m, video_dz, nz);
        for (int iz = iz_lo; iz <= iz_hi; ++iz) {
            amrex::Real const bin_z_lo =
                z_min_m + static_cast<amrex::Real>(iz) * video_dz;
            amrex::Real const overlap_z_lo = amrex::max(cell_z_lo, bin_z_lo);
            amrex::Real const overlap_z_hi =
                amrex::min(cell_z_hi, bin_z_lo + video_dz);
            for (int ir = ir_lo; ir <= ir_hi; ++ir) {
                amrex::Real const bin_r_lo =
                    r_min_m + static_cast<amrex::Real>(ir) * video_dr;
                amrex::Real const overlap_volume = rz_overlap_volume(
                    amrex::max(cell_r_lo, bin_r_lo),
                    amrex::min(cell_r_hi, bin_r_lo + video_dr),
                    overlap_z_lo,
                    overlap_z_hi);
                if (overlap_volume <= amrex::Real(0.0)) {
                    continue;
                }
                std::size_t const out_index =
                    static_cast<std::size_t>(iz) * nr + ir;
                weighted[out_index] += value * overlap_volume;
                volume[out_index] += overlap_volume;
            }
        }
    };

    auto const& positive_ion_mf = m_advance->Fluid(0).PositiveIonDensity();
    for (amrex::MFIter mfi(positive_ion_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const npos_arr = positive_ion_mf.const_array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) noexcept {
            rebin_cell(i, j, npos_arr(i, j, k),
                       positive_ion_density_volume_sum, positive_ion_volume);
        });
    }

    auto const& er_mf = *binding.efield_x;
    auto const& ez_mf = *binding.efield_z;
    for (amrex::MFIter mfi(ez_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const ez_arr = ez_mf.const_array(mfi);
        auto const er_arr = er_mf.const_array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) noexcept {
            amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
            amrex::Real const er = er_mf.boxArray()[mfi.index()].contains(iv)
                ? er_arr(i, j, k)
                : amrex::Real(0.0);
            amrex::Real const ez = ez_arr(i, j, k);
            amrex::Real const abs_e = std::sqrt(er * er + ez * ez);
            auto const [r, z] = cell_center(i, j);
            if (r >= r_min_m && r <= r_max_m && z >= active_z_min && z <= active_z_max) {
                amrex::Real const volume = cell_volume_rz(binding.geom, i, j);
                amrex::Real const abs_er = std::abs(er);
                amrex::Real const abs_ez = std::abs(ez);
                amrex::Real const er_over_ez =
                    abs_er / amrex::max(abs_ez, amrex::Real(1.0e-30));
                field_moments.volume += volume;
                field_moments.abs_er_volume_sum += volume * abs_er;
                field_moments.abs_ez_volume_sum += volume * abs_ez;
                field_moments.abs_e_volume_sum += volume * abs_e;
                field_moments.er_over_ez_volume_sum += volume * er_over_ez;
                field_moments.max_abs_er = amrex::max(field_moments.max_abs_er, abs_er);
                field_moments.max_abs_ez = amrex::max(field_moments.max_abs_ez, abs_ez);
                field_moments.max_abs_e = amrex::max(field_moments.max_abs_e, abs_e);
                field_moments.max_er_over_ez =
                    amrex::max(field_moments.max_er_over_ez, er_over_ez);
            }
            rebin_cell(i, j, abs_e / amrex::Real(1000.0),
                       field_volume_sum, field_volume);
        });
    }

    global_vector_sum(electron_weight);
    global_vector_sum(photon_weight_grid);
    global_vector_sum(positron_weight_grid);
    global_vector_sum(positive_ion_density_volume_sum);
    global_vector_sum(positive_ion_volume);
    global_vector_sum(field_volume_sum);
    global_vector_sum(field_volume);
    // Batch the frame moment scalars into one
    // array allreduce per op type instead of ~28 scalar allreduces per frame.
    particle_moments.macro_count = global_count_sum(particle_moments.macro_count);
    {
        std::vector<amrex::Real> moment_sums = {
            particle_moments.weight,
            particle_moments.r_sum,
            particle_moments.r2_sum,
            particle_moments.z_sum,
            particle_moments.z2_sum,
            particle_moments.kinetic_sum_eV,
            particle_moments.uz_over_u_sum,
            particle_moments.radial_over_u_sum,
            particle_moments.abs_ur_over_abs_uz_sum,
            particle_moments.positive_uz_weight,
            particle_moments.negative_uz_weight,
            field_moments.volume,
            field_moments.abs_er_volume_sum,
            field_moments.abs_ez_volume_sum,
            field_moments.abs_e_volume_sum,
            field_moments.er_over_ez_volume_sum};
        global_vector_sum(moment_sums);
        std::size_t moment_index = 0;
        particle_moments.weight = moment_sums[moment_index++];
        particle_moments.r_sum = moment_sums[moment_index++];
        particle_moments.r2_sum = moment_sums[moment_index++];
        particle_moments.z_sum = moment_sums[moment_index++];
        particle_moments.z2_sum = moment_sums[moment_index++];
        particle_moments.kinetic_sum_eV = moment_sums[moment_index++];
        particle_moments.uz_over_u_sum = moment_sums[moment_index++];
        particle_moments.radial_over_u_sum = moment_sums[moment_index++];
        particle_moments.abs_ur_over_abs_uz_sum = moment_sums[moment_index++];
        particle_moments.positive_uz_weight = moment_sums[moment_index++];
        particle_moments.negative_uz_weight = moment_sums[moment_index++];
        field_moments.volume = moment_sums[moment_index++];
        field_moments.abs_er_volume_sum = moment_sums[moment_index++];
        field_moments.abs_ez_volume_sum = moment_sums[moment_index++];
        field_moments.abs_e_volume_sum = moment_sums[moment_index++];
        field_moments.er_over_ez_volume_sum = moment_sums[moment_index++];
        std::vector<amrex::Real> moment_maxes = {
            particle_moments.max_r,
            particle_moments.max_z,
            field_moments.max_abs_er,
            field_moments.max_abs_ez,
            field_moments.max_abs_e,
            field_moments.max_er_over_ez};
        global_vector_max(moment_maxes);
        particle_moments.max_r = moment_maxes[0];
        particle_moments.max_z = moment_maxes[1];
        field_moments.max_abs_er = moment_maxes[2];
        field_moments.max_abs_ez = moment_maxes[3];
        field_moments.max_abs_e = moment_maxes[4];
        field_moments.max_er_over_ez = moment_maxes[5];
    }
    particle_moments.min_r = global_real_min(particle_moments.min_r);
    particle_moments.min_z = global_real_min(particle_moments.min_z);
    if (particle_moments.weight <= amrex::Real(0.0)) {
        particle_moments.min_r = amrex::Real(0.0);
        particle_moments.max_r = amrex::Real(0.0);
        particle_moments.min_z = amrex::Real(0.0);
        particle_moments.max_z = amrex::Real(0.0);
    }

    std::vector<amrex::Real> electron_density(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> photon_density(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> positron_density(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> positive_ion_density(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> abs_e_kvpm(cell_count, amrex::Real(0.0));
    amrex::Real total_energetic_weight = amrex::Real(0.0);
    amrex::Real total_photon_weight = amrex::Real(0.0);
    amrex::Real total_positron_weight = amrex::Real(0.0);
    amrex::Real max_density = amrex::Real(0.0);
    amrex::Real max_photon_density = amrex::Real(0.0);
    amrex::Real max_positron_density = amrex::Real(0.0);
    amrex::Real max_positive_ion_density = amrex::Real(0.0);
    amrex::Real min_abs_e = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real max_abs_e = amrex::Real(0.0);
    for (int iz = 0; iz < nz; ++iz) {
        for (int ir = 0; ir < nr; ++ir) {
            std::size_t const index = static_cast<std::size_t>(iz) * nr + ir;
            amrex::Real const bin_volume =
                video_bin_volume_rz(ir, nr, nz, r_min_m, r_max_m, z_min_m, z_max_m);
            if (bin_volume > amrex::Real(0.0)) {
                electron_density[index] = electron_weight[index] / bin_volume;
                photon_density[index] = photon_weight_grid[index] / bin_volume;
                positron_density[index] =
                    positron_weight_grid[index] / bin_volume;
            }
            if (positive_ion_volume[index] > amrex::Real(0.0)) {
                positive_ion_density[index] =
                    positive_ion_density_volume_sum[index] / positive_ion_volume[index];
            }
            if (field_volume[index] > amrex::Real(0.0)) {
                abs_e_kvpm[index] = field_volume_sum[index] / field_volume[index];
            }
            total_energetic_weight += electron_weight[index];
            total_photon_weight += photon_weight_grid[index];
            total_positron_weight += positron_weight_grid[index];
            max_density = amrex::max(max_density, electron_density[index]);
            max_photon_density =
                amrex::max(max_photon_density, photon_density[index]);
            max_positron_density =
                amrex::max(max_positron_density, positron_density[index]);
            max_positive_ion_density =
                amrex::max(max_positive_ion_density, positive_ion_density[index]);
            if (abs_e_kvpm[index] > amrex::Real(0.0)) {
                min_abs_e = amrex::min(min_abs_e, abs_e_kvpm[index]);
                max_abs_e = amrex::max(max_abs_e, abs_e_kvpm[index]);
            }
        }
    }
    if (!std::isfinite(static_cast<double>(min_abs_e))) {
        min_abs_e = amrex::Real(0.0);
    }

    while (m_next_video_frame < m_video_frames.size()
           && m_video_frames[m_next_video_frame].step == step) {
        auto const request = m_video_frames[m_next_video_frame];
        if (!m_video_stream_initialized && m_video_frame_write_count > 0) {
            // Restarted run: the checkpoint restored an already-advanced frame
            // counter, so the stream files hold earlier frames.  Re-running
            // the truncating initialization would destroy them; verify the
            // binary stream is present and resume in append mode instead.
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ifstream existing(
                    join_path(m_video_diag_output_dir, "video_frames.bin"),
                    std::ios::binary);
                if (!existing) {
                    amrex::Abort(
                        "RREA video restart: video_frames.bin is missing from "
                        + m_video_diag_output_dir
                        + " although the checkpoint records "
                        + std::to_string(m_video_frame_write_count)
                        + " already-written frames");
                }
            }
            m_video_stream_initialized = true;
        }
        if (!m_video_stream_initialized) {
            collective_create_dir_all(m_video_diag_output_dir);
            if (amrex::ParallelDescriptor::IOProcessor()) {
                {
                    std::ofstream frames(
                        join_path(m_video_diag_output_dir, "video_frames.bin"),
                        std::ios::binary | std::ios::trunc);
                    if (!frames) {
                        amrex::FileOpenFailed(
                            join_path(m_video_diag_output_dir, "video_frames.bin"));
                    }
                }
                {
                    std::ofstream metadata(
                        join_path(m_video_diag_output_dir, "video_frame_metadata.csv"),
                        std::ios::trunc);
                    if (!metadata) {
                        amrex::FileOpenFailed(
                            join_path(m_video_diag_output_dir, "video_frame_metadata.csv"));
                    }
                    metadata
                        << "write_index,capture_index,step,time_s,target_time_s,nr,nz,"
                        << "r_min_m,r_max_m,z_min_m,z_max_m,altitude_msl_at_z0_m,"
                        << "energy_threshold_eV,total_energetic_weight,"
                        << "max_energetic_density_m3,max_positive_ion_density_m3,"
                        << "min_abs_E_kVpm,max_abs_E_kVpm,"
                        << "total_photon_ge_threshold_weight,"
                        << "max_photon_ge_threshold_density_m3,"
                        << "total_positron_ge_threshold_weight,"
                        << "max_positron_ge_threshold_density_m3,"
                        // video-v4 frames are variable length, so these two
                        // columns ARE the frame index: the reader seeks here
                        // and the restart trim truncates to
                        // offset+length of the last checkpointed frame.
                        << "byte_offset,byte_length\n";
                }
                {
                    std::ofstream particles(
                        join_path(m_video_diag_output_dir, "particle_moments.csv"),
                        std::ios::trunc);
                    if (!particles) {
                        amrex::FileOpenFailed(
                            join_path(m_video_diag_output_dir, "particle_moments.csv"));
                    }
                    particles
                        << "write_index,capture_index,step,time_s,target_time_s,"
                        << "energetic_macro_count,energetic_weight,"
                        << "mean_r_m,rms_r_m,min_r_m,max_r_m,"
                        << "mean_z_m,rms_z_m,min_z_m,max_z_m,"
                        << "mean_kinetic_energy_eV,mean_uz_over_u,"
                        << "mean_radial_u_over_u,mean_abs_radial_over_abs_z,"
                        << "fraction_uz_positive,fraction_uz_negative\n";
                }
                {
                    std::ofstream fields(
                        join_path(m_video_diag_output_dir, "field_direction_moments.csv"),
                        std::ios::trunc);
                    if (!fields) {
                        amrex::FileOpenFailed(
                            join_path(m_video_diag_output_dir, "field_direction_moments.csv"));
                    }
                    fields
                        << "write_index,capture_index,step,time_s,target_time_s,"
                        << "active_z_min_m,active_z_max_m,volume_m3,"
                        << "mean_abs_Er_kVpm,mean_abs_Ez_kVpm,mean_abs_E_kVpm,"
                        << "mean_abs_Er_over_abs_Ez,max_abs_Er_kVpm,"
                        << "max_abs_Ez_kVpm,max_abs_E_kVpm,max_abs_Er_over_abs_Ez,"
                        << "max_abs_E_over_E0_peak\n";
                }
                {
                    std::ofstream summary(
                        join_path(m_video_diag_output_dir, "video_capture_metadata.json"),
                        std::ios::trunc);
                    if (!summary) {
                        amrex::FileOpenFailed(
                            join_path(m_video_diag_output_dir, "video_capture_metadata.json"));
                    }
                    summary << std::setprecision(17)
                            << "{\n"
                            << "  \"format\": \"rrea_video_frames_binary_v4\",\n"
                            << "  \"array_order\": \"z_major_r_minor\",\n"
                            // Photon/positron
                            // population grids appended (append-only order so
                            // name-based readers of the v2 arrays are
                            // unaffected; the renderer derives its frame
                            // stride from this list).
                            << "  \"arrays_per_frame\": [\"n_energetic_e_ge_threshold_m3\", \"n_positive_ion_m3\", \"abs_E_kVpm\", \"n_photon_ge_threshold_m3\", \"n_positron_ge_threshold_m3\"],\n"
                            // The frame payload is float32 regardless of the
                            // build's amrex::Real: v4 narrows on write.
                            << "  \"real_size_bytes\": "
                            << rrea::video::frame_codec_itemsize << ",\n"
                            << "  \"nr\": " << nr << ",\n"
                            << "  \"nz\": " << nz << ",\n"
                            << "  \"r_min_m\": " << r_min_m << ",\n"
                            << "  \"r_max_m\": " << r_max_m << ",\n"
                            << "  \"z_min_m\": " << z_min_m << ",\n"
                            << "  \"z_max_m\": " << z_max_m << ",\n"
                            << "  \"altitude_msl_at_z0_m\": " << m_profile_altitude_msl_at_z0_m << ",\n"
                            << "  \"field_taper_r_start_m\": "
                            << m_advance_config.field_initializer.taper_r_start_m << ",\n"
                            << "  \"field_taper_r_end_m\": "
                            << m_advance_config.field_initializer.taper_r_end_m << ",\n"
                            << "  \"energy_threshold_eV\": " << m_video_diag_energy_threshold_eV << ",\n"
                            << "  \"density_profile_path\": \""
                            << json_escape(m_density_profile_path) << "\",\n"
                            << "  \"transport_density_profile_enabled\": "
                            << (m_transport_density_profile_enabled ? 1 : 0) << ",\n"
                            << "  \"transport_density_ratio\": "
                            << m_transport_density_ratio << ",\n"
                            << "  \"ion_drift_enable\": "
                            << (m_advance_config.fluid.ion_drift_enable ? 1 : 0) << ",\n"
                            << "  \"ion_mobility_model_input\": \""
                            << json_escape(m_ion_mobility_model) << "\",\n"
                            << "  \"cxx_low_energy_mobility_model_effective\": \""
                            << json_escape(m_advance_config.fluid.ion_mobility_model)
                            << "\",\n"
                            << "  \"positive_ion_mobility_m2_per_vs\": "
                            << m_advance_config.fluid.positive_ion_mobility_m2_per_vs << ",\n"
                            << "  \"negative_ion_mobility_m2_per_vs\": "
                            << m_advance_config.fluid.negative_ion_mobility_m2_per_vs << ",\n"
                            << "  \"positive_ion_reduced_mobility_stp_m2_per_vs\": "
                            << m_advance_config.fluid.positive_ion_reduced_mobility_stp_m2_per_vs << ",\n"
                            << "  \"negative_ion_reduced_mobility_stp_m2_per_vs\": "
                            << m_advance_config.fluid.negative_ion_reduced_mobility_stp_m2_per_vs << ",\n"
                            << "  \"ion_mobility_density_ratio_floor\": "
                            << m_advance_config.fluid.ion_mobility_density_ratio_floor << ",\n"
                            << "  \"ion_drift_cfl\": "
                            << m_advance_config.fluid.ion_drift_cfl << ",\n"
                            << "  \"frame_schedule_file\": \""
                            << json_escape(m_video_diag_frame_schedule_path) << "\",\n"
                            << "  \"case_id\": \"" << json_escape(m_case_id) << "\"\n"
                            << "}\n";
                }
            }
            m_video_stream_initialized = true;
        }

        std::uint64_t frame_byte_offset = 0;
        std::uint64_t frame_byte_length = 0;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            {
                // video-v4: one shuffled, deflated float32 blob per frame
                // (rrea::video::EncodeFrame states the encoding).  The offset
                // is taken from the stream BEFORE the write and the length
                // after, so the two metadata columns index this exact blob
                // even if a previous restart trimmed the file.
                std::vector<unsigned char> blob;
                rrea::video::EncodeFrame(
                    {electron_density.data(), positive_ion_density.data(),
                     abs_e_kvpm.data(), photon_density.data(),
                     positron_density.data()},
                    electron_density.size(), blob);

                std::ofstream frames(
                    join_path(m_video_diag_output_dir, "video_frames.bin"),
                    std::ios::binary | std::ios::app);
                if (!frames) {
                    amrex::FileOpenFailed(join_path(m_video_diag_output_dir, "video_frames.bin"));
                }
                frame_byte_offset =
                    static_cast<std::uint64_t>(frames.tellp());
                frames.write(
                    reinterpret_cast<char const*>(blob.data()),
                    static_cast<std::streamsize>(blob.size()));
                if (!frames) {
                    amrex::Abort(
                        "rrea video-v4: short write appending a frame to "
                        "video_frames.bin -- every later byte_offset would be "
                        "wrong, so refusing to continue");
                }
                frame_byte_length = static_cast<std::uint64_t>(blob.size());
            }
            {
                std::ofstream metadata(
                    join_path(m_video_diag_output_dir, "video_frame_metadata.csv"),
                    std::ios::app);
                if (!metadata) {
                    amrex::FileOpenFailed(
                        join_path(m_video_diag_output_dir, "video_frame_metadata.csv"));
                }
                metadata << std::setprecision(17)
                         << m_video_frame_write_count << ","
                         << request.capture_index << ","
                         << step << ","
                         << time_s << ","
                         << request.target_time_s << ","
                         << nr << ","
                         << nz << ","
                         << r_min_m << ","
                         << r_max_m << ","
                         << z_min_m << ","
                         << z_max_m << ","
                         << m_profile_altitude_msl_at_z0_m << ","
                         << m_video_diag_energy_threshold_eV << ","
                         << total_energetic_weight << ","
                         << max_density << ","
                         << max_positive_ion_density << ","
                         << min_abs_e << ","
                         << max_abs_e << ","
                         << total_photon_weight << ","
                         << max_photon_density << ","
                         << total_positron_weight << ","
                         << max_positron_density << ","
                         << frame_byte_offset << ","
                         << frame_byte_length << "\n";
            }
            {
                amrex::Real const inv_weight =
                    particle_moments.weight > amrex::Real(0.0)
                    ? amrex::Real(1.0) / particle_moments.weight
                    : amrex::Real(0.0);
                amrex::Real const mean_r = particle_moments.r_sum * inv_weight;
                amrex::Real const mean_z = particle_moments.z_sum * inv_weight;
                amrex::Real const rms_r =
                    particle_moments.weight > amrex::Real(0.0)
                    ? std::sqrt(amrex::max(
                        particle_moments.r2_sum * inv_weight,
                        amrex::Real(0.0)))
                    : amrex::Real(0.0);
                amrex::Real const rms_z =
                    particle_moments.weight > amrex::Real(0.0)
                    ? std::sqrt(amrex::max(
                        particle_moments.z2_sum * inv_weight,
                        amrex::Real(0.0)))
                    : amrex::Real(0.0);
                std::ofstream particles(
                    join_path(m_video_diag_output_dir, "particle_moments.csv"),
                    std::ios::app);
                if (!particles) {
                    amrex::FileOpenFailed(
                        join_path(m_video_diag_output_dir, "particle_moments.csv"));
                }
                particles << std::setprecision(17)
                          << m_video_frame_write_count << ","
                          << request.capture_index << ","
                          << step << ","
                          << time_s << ","
                          << request.target_time_s << ","
                          << particle_moments.macro_count << ","
                          << particle_moments.weight << ","
                          << mean_r << ","
                          << rms_r << ","
                          << particle_moments.min_r << ","
                          << particle_moments.max_r << ","
                          << mean_z << ","
                          << rms_z << ","
                          << particle_moments.min_z << ","
                          << particle_moments.max_z << ","
                          << particle_moments.kinetic_sum_eV * inv_weight << ","
                          << particle_moments.uz_over_u_sum * inv_weight << ","
                          << particle_moments.radial_over_u_sum * inv_weight << ","
                          << particle_moments.abs_ur_over_abs_uz_sum * inv_weight << ","
                          << particle_moments.positive_uz_weight * inv_weight << ","
                          << particle_moments.negative_uz_weight * inv_weight << "\n";
            }
            {
                amrex::Real const inv_volume =
                    field_moments.volume > amrex::Real(0.0)
                    ? amrex::Real(1.0) / field_moments.volume
                    : amrex::Real(0.0);
                amrex::Real const e0_peak =
                    std::abs(m_advance_config.field_initializer.ez_acceleration_v_per_m);
                std::ofstream fields(
                    join_path(m_video_diag_output_dir, "field_direction_moments.csv"),
                    std::ios::app);
                if (!fields) {
                    amrex::FileOpenFailed(
                        join_path(m_video_diag_output_dir, "field_direction_moments.csv"));
                }
                fields << std::setprecision(17)
                       << m_video_frame_write_count << ","
                       << request.capture_index << ","
                       << step << ","
                       << time_s << ","
                       << request.target_time_s << ","
                       << active_z_min << ","
                       << active_z_max << ","
                       << field_moments.volume << ","
                       << field_moments.abs_er_volume_sum * inv_volume / amrex::Real(1000.0) << ","
                       << field_moments.abs_ez_volume_sum * inv_volume / amrex::Real(1000.0) << ","
                       << field_moments.abs_e_volume_sum * inv_volume / amrex::Real(1000.0) << ","
                       << field_moments.er_over_ez_volume_sum * inv_volume << ","
                       << field_moments.max_abs_er / amrex::Real(1000.0) << ","
                       << field_moments.max_abs_ez / amrex::Real(1000.0) << ","
                       << field_moments.max_abs_e / amrex::Real(1000.0) << ","
                       << field_moments.max_er_over_ez << ","
                       << (e0_peak > amrex::Real(0.0)
                           ? field_moments.max_abs_e / e0_peak
                           : amrex::Real(0.0))
                       << "\n";
            }
        }
        ++m_video_frame_write_count;
        ++m_next_video_frame;
    }
}

}  // namespace rrea::warpx
