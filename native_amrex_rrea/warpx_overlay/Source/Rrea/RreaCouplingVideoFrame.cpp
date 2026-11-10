// Video-frame capture for RreaWarpXCoupling: the frame schedule, the per-frame
// species/ion/field grids, the energetic-electron spatial moments, and the
// video-v4 stream writer.

#include "RreaWarpXCoupling.H"

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
using detail::global_vector_sum;
using detail::join_path;

namespace {

using rrea::cell_index_origin;
using rrea::csv_text::split_csv_line;

struct VideoParticleMoments {
    std::uint64_t macro_count = 0;
    amrex::Real weight = amrex::Real(0.0);
    amrex::Real r_sum = amrex::Real(0.0);
    amrex::Real r2_sum = amrex::Real(0.0);
    amrex::Real z_sum = amrex::Real(0.0);
    amrex::Real z2_sum = amrex::Real(0.0);
    amrex::Real min_z = std::numeric_limits<amrex::Real>::infinity();
    amrex::Real max_z = -std::numeric_limits<amrex::Real>::infinity();
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
    std::ifstream input(m_video_diag_frame_schedule_path);
    if (!input) {
        amrex::FileOpenFailed(m_video_diag_frame_schedule_path);
    }
    std::string header;
    std::getline(input, header);
    auto const columns = split_csv_line(header);
    auto column = [&](char const* name) {
        auto const found = std::find(columns.begin(), columns.end(), name);
        if (found == columns.end()) {
            amrex::Abort(std::string("RREA video frame schedule lacks column ")
                         + name + ": " + m_video_diag_frame_schedule_path);
        }
        return static_cast<std::size_t>(found - columns.begin());
    };
    std::size_t const capture_index_col = column("capture_index");
    std::size_t const step_col = column("step");
    std::size_t const time_col = column("target_time_s");

    m_video_frames.clear();
    std::string line;
    while (std::getline(input, line)) {
        if (rrea::csv_text::trim(line).empty()) {
            continue;
        }
        auto const fields = split_csv_line(line);
        if (fields.size() != columns.size()) {
            amrex::Abort("RREA video frame schedule row does not match its header: "
                         + line);
        }
        RreaVideoFrameRequest request;
        request.capture_index = std::stoi(fields[capture_index_col]);
        request.step = std::stoi(fields[step_col]);
        request.target_time_s =
            static_cast<amrex::Real>(std::stod(fields[time_col]));
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
    if (!m_video_diag_enable || !m_initialized) {
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

    auto const& binding = m_level;
    auto const prob_lo = binding.geom.ProbLoArray();
    auto const prob_hi = binding.geom.ProbHiArray();
    amrex::Real const domain_r_min_m = prob_lo[0];
    amrex::Real const domain_r_max_m = prob_hi[0];
    amrex::Real const domain_z_min_m = prob_lo[1];
    amrex::Real const domain_z_max_m = prob_hi[1];
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
    amrex::Real const video_dr = (r_max_m - r_min_m) / static_cast<amrex::Real>(nr);
    amrex::Real const video_dz = (z_max_m - z_min_m) / static_cast<amrex::Real>(nz);

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
            if (p.KineticEnergyEv() < m_video_diag_energy_threshold_eV) {
                return;
            }
            amrex::Real const particle_weight = p.weight;
            ++particle_moments.macro_count;
            particle_moments.weight += particle_weight;
            particle_moments.r_sum += particle_weight * p.r;
            particle_moments.r2_sum += particle_weight * p.r * p.r;
            particle_moments.z_sum += particle_weight * p.z;
            particle_moments.z2_sum += particle_weight * p.z * p.z;
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
        return std::make_pair(
            plo[0] + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) * dx[0],
            plo[1] + (static_cast<amrex::Real>(j) + amrex::Real(0.5)) * dx[1]);
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
        amrex::Real const cell_z_lo = z - amrex::Real(0.5) * dx[1];
        amrex::Real const cell_z_hi = z + amrex::Real(0.5) * dx[1];
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

    auto const& positive_ion_mf = m_advance->Fluid().PositiveIonDensity();
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
            amrex::Real const er = er_arr(i, j, k);
            amrex::Real const ez = ez_arr(i, j, k);
            rebin_cell(i, j, std::sqrt(er * er + ez * ez) / amrex::Real(1000.0),
                       field_volume_sum, field_volume);
        });
    }

    global_vector_sum(electron_weight);
    auto creation_weights = m_electron_creation.weights;
    if (!creation_weights.empty()) global_vector_sum(creation_weights);
    global_vector_sum(photon_weight_grid);
    global_vector_sum(positron_weight_grid);
    global_vector_sum(positive_ion_density_volume_sum);
    global_vector_sum(positive_ion_volume);
    global_vector_sum(field_volume_sum);
    global_vector_sum(field_volume);
    particle_moments.macro_count = global_count_sum(particle_moments.macro_count);
    {
        std::vector<amrex::Real> moment_sums = {
            particle_moments.weight,
            particle_moments.r_sum,
            particle_moments.r2_sum,
            particle_moments.z_sum,
            particle_moments.z2_sum};
        global_vector_sum(moment_sums);
        particle_moments.weight = moment_sums[0];
        particle_moments.r_sum = moment_sums[1];
        particle_moments.r2_sum = moment_sums[2];
        particle_moments.z_sum = moment_sums[3];
        particle_moments.z2_sum = moment_sums[4];
    }
    amrex::ParallelDescriptor::ReduceRealMin(particle_moments.min_z);
    amrex::ParallelDescriptor::ReduceRealMax(particle_moments.max_z);
    if (particle_moments.weight <= amrex::Real(0.0)) {
        particle_moments.min_z = amrex::Real(0.0);
        particle_moments.max_z = amrex::Real(0.0);
    }

    std::vector<amrex::Real> electron_density(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> photon_density(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> positron_density(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> positive_ion_density(cell_count, amrex::Real(0.0));
    std::vector<amrex::Real> abs_e_kvpm(cell_count, amrex::Real(0.0));
    amrex::Real total_energetic_weight = amrex::Real(0.0);
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
        }
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
                if (!creation_weights.empty()) {
                    std::ofstream history(join_path(m_video_diag_output_dir,
                        "electron_creation_history.bin.gz"),std::ios::binary);
                    if (!history) amrex::Abort("cannot initialize electron creation history");
                }
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
                        << "write_index,capture_index,step,time_s,target_time_s,"
                        << "total_energetic_weight,"
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
                        << "mean_r_m,rms_r_m,mean_z_m,rms_z_m,min_z_m,max_z_m\n";
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
                            << "  \"arrays_per_frame\": [\"n_energetic_e_ge_threshold_m3\", \"n_positive_ion_m3\", \"abs_E_kVpm\", \"n_photon_ge_threshold_m3\", \"n_positron_ge_threshold_m3\"],\n"
                            << "  \"nr\": " << nr << ",\n"
                            << "  \"nz\": " << nz << ",\n"
                            << "  \"r_min_m\": " << r_min_m << ",\n"
                            << "  \"r_max_m\": " << r_max_m << ",\n"
                            << "  \"z_min_m\": " << z_min_m << ",\n"
                            << "  \"z_max_m\": " << z_max_m << ",\n"
                            << "  \"altitude_msl_at_z0_m\": " << m_profile_altitude_msl_at_z0_m << "\n"
                            << "}\n";
                }
            }
            m_video_stream_initialized = true;
        }

        std::uint64_t frame_byte_offset = 0;
        std::uint64_t frame_byte_length = 0;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (!creation_weights.empty()) {
                auto const history = join_path(m_video_diag_output_dir,"electron_creation_history.bin.gz");
                m_electron_creation.Write(history,time_s,m_profile_altitude_msl_at_z0_m,
                                          creation_weights,true);
                m_electron_creation.history_bytes = std::filesystem::file_size(history);
            }
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
                         << total_energetic_weight << ","
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
                // Axial spread about the centroid; the radial rms is about the axis.
                amrex::Real const rms_z =
                    particle_moments.weight > amrex::Real(0.0)
                    ? std::sqrt(amrex::max(
                        particle_moments.z2_sum * inv_weight - mean_z * mean_z,
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
                          << mean_z << ","
                          << rms_z << ","
                          << particle_moments.min_z << ","
                          << particle_moments.max_z << "\n";
            }
        }
        ++m_video_frame_write_count;
        ++m_next_video_frame;
    }
}

}  // namespace rrea::warpx
