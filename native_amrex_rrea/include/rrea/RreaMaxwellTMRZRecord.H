#ifndef RREA_MAXWELL_TM_RZ_RECORD_H
#define RREA_MAXWELL_TM_RZ_RECORD_H

// Persistence of the Maxwell TM RZ state: the checkpointed scattered field
// and the EM near-to-far Huygens record.  No physics lives here; the solver
// is RreaMaxwellTMRZ.H.  Rank 0 writes; every rank reads the identical
// replicated state.

#include "rrea/RreaMaxwellTMRZ.H"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rrea {

// Scattered Er/Ez/Btheta only: the ambient is analytic and rebuilt on every
// start, so it is never written.  One trailing double carries the run's
// cumulative signed continuity residual (units e); a file written before it
// existed simply ends after Btheta.  The reader owns the mesh check.
inline void WriteMaxwellState(
    MaxwellTMRZ const& f, std::string const& dir, double continuity_signed_e)
{
    GpuSynchronize();
    std::ofstream out(dir + "/maxwell_tm_state.bin", std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write Maxwell TM state to " + dir);
    }
    std::int64_t const header[2] = {f.nr, f.nz};
    out.write(reinterpret_cast<char const*>(header), sizeof(header));
    for (auto const* values : {&f.er, &f.ez, &f.btheta}) {
        out.write(reinterpret_cast<char const*>(values->data()),
                  static_cast<std::streamsize>(
                      values->size() * sizeof(double)));
    }
    out.write(reinterpret_cast<char const*>(&continuity_signed_e),
              sizeof(continuity_signed_e));
}

// Returns false when the file predates the continuity accumulator.
inline bool ReadMaxwellState(
    MaxwellTMRZ& f, std::string const& dir, double& continuity_signed_e)
{
    GpuSynchronize();
    std::ifstream in(dir + "/maxwell_tm_state.bin", std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "Maxwell TM state is missing from checkpoint " + dir);
    }
    std::int64_t header[2] = {0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (header[0] != f.nr || header[1] != f.nz) {
        throw std::runtime_error(
            "Maxwell TM state mesh does not match this run's domain");
    }
    for (auto* values : {&f.er, &f.ez, &f.btheta}) {
        in.read(reinterpret_cast<char*>(values->data()),
                static_cast<std::streamsize>(values->size() * sizeof(double)));
    }
    if (!in) {
        throw std::runtime_error(
            "Maxwell TM state in " + dir + " is truncated");
    }
    double value = 0.0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(value))) {
        return false;
    }
    continuity_signed_e = value;
    return true;
}

// EM near-to-far Huygens record: per call, one binary frame of the SCATTERED
// fields on a closed surface (the static ambient does not radiate).
//
// The surface sits kRreaEmSurfaceMarginCells inside the open boundaries
// (rationale at that constant).
//
// Version 4 labels E at the completed step time and Btheta at time - dt/2.
//
// Frame layout (all doubles), with (ns_r, ns_z) the SURFACE dimensions and
// indices local to the surface box:
//   time_s,
//   top    z_top: Er_surface(i) i=0..ns_r | Ez(i) i=0..ns_r-1 | Bt(i) i=0..ns_r
//   bottom z_bot: same three, on the lower face
//   side   r_side: Er(j) j=0..ns_z-1 | Ez_surface(j) j=0..ns_z | Bt(j) j=0..ns_z
// Header: 4 int64 (magic 0x52454D42, version, ns_r, ns_z) + 3 doubles
// (dr, dz, z of the surface's lower face).  The analysis therefore needs no
// knowledge of the margin: the header already describes the surface itself.
// Restart appends require the same physical surface and version; the first
// record of a restart first drops every frame at or after its own time.
inline void AppendMaxwellBoundaryRecord(
    MaxwellTMRZ const& f, std::string const& dir, double time_s,
    bool first_record = false)
{
    GpuSynchronize();
    int const margin = std::min(
        kRreaEmSurfaceMarginCells, std::min(f.nr / 4, f.nz / 4));
    int const i_side = f.nr - margin;
    int const j_bot = margin;
    int const j_top = f.nz - margin;
    int const ns_r = i_side;
    int const ns_z = j_top - j_bot;
    if (ns_r < 2 || ns_z < 2) {
        throw std::runtime_error(
            "EM Huygens surface does not fit inside this mesh");
    }

    std::string const path = dir + "/rrea_em_boundary.bin";
    bool const fresh = !std::ifstream(path).good();
    std::int64_t constexpr record_version = 4;
    std::int64_t const header[4] = {0x52454D42, record_version, ns_r, ns_z};
    double const mesh[3] = {f.dr, f.dz, f.z_lo + j_bot * f.dz};
    std::uintmax_t constexpr header_bytes = sizeof(header) + sizeof(mesh);
    std::uintmax_t const frame_bytes = sizeof(double)
        * (1 + 4 * (ns_r + 1) + 2 * ns_r + ns_z + 2 * (ns_z + 1));
    if (!fresh) {
        std::ifstream existing(path, std::ios::binary);
        std::int64_t existing_header[4] = {0, 0, 0, 0};
        double existing_mesh[3] = {};
        existing.read(
            reinterpret_cast<char*>(existing_header), sizeof(existing_header));
        existing.read(reinterpret_cast<char*>(existing_mesh), sizeof(existing_mesh));
        if (!existing || !std::equal(std::begin(header), std::end(header), existing_header)
            || !std::equal(std::begin(mesh), std::end(mesh), existing_mesh)) {
            throw std::runtime_error(
                "existing EM boundary record is incompatible with this run: "
                + path);
        }
        auto const payload_end = std::filesystem::file_size(path);
        if (first_record) {
            double const tolerance = 32 * std::numeric_limits<double>::epsilon()
                * std::abs(time_s);
            std::uintmax_t end = header_bytes;
            for (; end + frame_bytes <= payload_end; end += frame_bytes) {
                double recorded_time = 0.0;
                existing.seekg(static_cast<std::streamoff>(end));
                existing.read(reinterpret_cast<char*>(&recorded_time), sizeof(recorded_time));
                if (!existing || !std::isfinite(recorded_time)) {
                    throw std::runtime_error("invalid EM record time: " + path);
                }
                if (recorded_time >= time_s - tolerance) { break; }
            }
            existing.close();
            std::filesystem::resize_file(path, end);
        } else if ((payload_end - header_bytes) % frame_bytes) {
            throw std::runtime_error("incomplete EM record; restart before appending: " + path);
        }
    }
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) {
        throw std::runtime_error("cannot append EM boundary record " + path);
    }
    if (fresh) {
        out.write(reinterpret_cast<char const*>(header), sizeof(header));
        out.write(reinterpret_cast<char const*>(mesh), sizeof(mesh));
    }

    // Tangential E straddles the surface: Er samples sit at z=(j+1/2)dz, so
    // the z faces average rows j-1 and j; Ez samples sit at r=(i+1/2)dr, so
    // the side face averages columns i-1 and i.  Normal E and Btheta already
    // lie exactly on the surface.
    auto er_on_z_face = [&f](int i, int j_face) {
        return 0.5 * (f.Er(i, j_face - 1) + f.Er(i, j_face));
    };
    auto ez_on_r_face = [&f](int i_face, int j) {
        return 0.5 * (f.Ez(i_face - 1, j) + f.Ez(i_face, j));
    };

    std::vector<double> frame;
    frame.reserve(1 + 4 * (ns_r + 1) + 2 * ns_r + ns_z + 2 * (ns_z + 1));
    frame.push_back(time_s);
    for (int i = 0; i <= ns_r; ++i) { frame.push_back(er_on_z_face(i, j_top)); }
    for (int i = 0; i < ns_r; ++i) { frame.push_back(f.Ez(i, j_top)); }
    for (int i = 0; i <= ns_r; ++i) { frame.push_back(f.Bt(i, j_top)); }
    for (int i = 0; i <= ns_r; ++i) { frame.push_back(er_on_z_face(i, j_bot)); }
    for (int i = 0; i < ns_r; ++i) { frame.push_back(f.Ez(i, j_bot)); }
    for (int i = 0; i <= ns_r; ++i) { frame.push_back(f.Bt(i, j_bot)); }
    for (int j = j_bot; j < j_top; ++j) { frame.push_back(f.Er(i_side, j)); }
    for (int j = j_bot; j <= j_top; ++j) {
        frame.push_back(ez_on_r_face(i_side, j));
    }
    for (int j = j_bot; j <= j_top; ++j) { frame.push_back(f.Bt(i_side, j)); }
    out.write(reinterpret_cast<char const*>(frame.data()),
              static_cast<std::streamsize>(frame.size() * sizeof(double)));
}

}  // namespace rrea

#endif  // RREA_MAXWELL_TM_RZ_RECORD_H
