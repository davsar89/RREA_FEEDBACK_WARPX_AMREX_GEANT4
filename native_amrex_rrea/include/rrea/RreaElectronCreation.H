#ifndef RREA_ELECTRON_CREATION_H
#define RREA_ELECTRON_CREATION_H
#include "rrea/RreaKineticSupport.H"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <zlib.h>

namespace rrea {
inline constexpr double electron_creation_threshold_eV = 1.0e6;
inline constexpr char electron_creation_attribute[] = "rrea_electron_counted";

// First passage of K >= 1 MeV: an electron contributes its weight once.
// Linear interpolation locates the crossing within a transport segment;
// equal endpoint energies also cover an electron born above threshold.
// The flag follows numerical splits and checkpoints, never physical daughters.
RREA_HOST_DEVICE inline double ElectronFirstPassage(
    bool& counted, double begin_eV, double end_eV) noexcept
{
    if (counted || (begin_eV < electron_creation_threshold_eV
                   && end_eV < electron_creation_threshold_eV)) return -1;
    counted = true;
    return begin_eV >= electron_creation_threshold_eV ? 0
        : (electron_creation_threshold_eV - begin_eV) / (end_eV - begin_eV);
}

// Undo the leading Strang half-kick and distribute its full work along the
// chord: K_phys(f) = K_split(f) + (f-1/2)*W. This avoids counting an
// intermediate half-kick overshoot when drag keeps the physical K below 1 MeV.
RREA_HOST_DEVICE inline double ElectronFirstPassageInDragSegment(
    bool& counted, double split_begin_eV, double drag_eV, double work_eV,
    double begin, double end, double chord_begin, double chord_end) noexcept
{
    double const scale = 1 / (chord_end-chord_begin);
    return ElectronFirstPassage(counted,
        split_begin_eV + ((begin-chord_begin)*scale-0.5)*work_eV,
        split_begin_eV-drag_eV + ((end-chord_begin)*scale-0.5)*work_eV);
}

// Cumulative weighted events, not occupancy: n_created = sum(w_first)/dV,
// dV = pi*(r_hi^2-r_lo^2)*dz. Keep weights here; the plot owns normalization.
// Channel 0: first passage inside the domain (including energetic births).
// Channel 1: externally injected electrons already above threshold.
struct ElectronCreationMap {
    std::array<std::int32_t, 2> shape{}; // nr, nz
    std::array<double, 4> extent{};     // r_lo, r_hi, z_lo, z_hi (m)
    std::vector<double> weights;
    std::uint64_t history_bytes = 0; // committed gzip-member boundary at checkpoint

    void Initialize(double bin_m, std::array<double, 4> bounds)
    {
        if (bin_m == 0) return;
        if (!std::isfinite(bin_m) || bin_m < 0)
            throw std::runtime_error("electron_creation_bin_m must be finite and nonnegative");
        extent = bounds;
        for (int d=0; d<2; ++d) {
            double const n = std::ceil((extent[2*d+1]-extent[2*d])/bin_m);
            if (!(n >= 1 && n <= 100000))
                throw std::runtime_error("invalid electron creation grid extent/resolution");
            shape[d] = static_cast<std::int32_t>(n);
        }
        weights.assign(2 * static_cast<std::size_t>(shape[0]) * shape[1], 0);
    }
    void Add(double r, double z, double weight, bool injected)
    {
        if (weights.empty()) return;
        if (!(r >= extent[0] && r < extent[1] && z >= extent[2] && z < extent[3]
              && std::isfinite(weight) && weight >= 0))
            throw std::runtime_error("invalid energetic electron creation event");
        auto const ir = static_cast<std::size_t>((r-extent[0])/(extent[1]-extent[0])*shape[0]);
        auto const iz = static_cast<std::size_t>((z-extent[2])/(extent[3]-extent[2])*shape[1]);
        auto const i = (static_cast<std::size_t>(injected)*shape[1]+iz)*shape[0]+ir;
        weights[i] += weight;
    }
    // One global grid in the checkpoint, independent of MPI layout. Native
    // little-endian binary64, as in the project's other checkpoint streams.
    void Write(std::string const& path, double time_s, double altitude_m,
               std::vector<double> const& global, bool append_gzip = false) const
    {
        std::ostringstream out(std::ios::binary);
        std::array<double,3> const info{electron_creation_threshold_eV,time_s,altitude_m};
        out.write("RREACR2\0",8);
        out.write(reinterpret_cast<char const*>(shape.data()),sizeof(shape));
        out.write(reinterpret_cast<char const*>(extent.data()),sizeof(extent));
        out.write(reinterpret_cast<char const*>(info.data()),sizeof(info));
        out.write(reinterpret_cast<char const*>(&history_bytes),sizeof(history_bytes));
        out.write(reinterpret_cast<char const*>(global.data()),global.size()*sizeof(double));
        auto const bytes = out.str();
        if (append_gzip) {
            gzFile file = gzopen(path.c_str(), "ab6");
            if (!file) throw std::runtime_error("cannot append electron creation history: " + path);
            bool ok = true;
            for (std::size_t offset=0; offset<bytes.size();) {
                auto const n = static_cast<unsigned int>(std::min(bytes.size()-offset, std::size_t(1)<<30));
                if (gzwrite(file, bytes.data()+offset, n) != static_cast<int>(n)) { ok=false; break; }
                offset += n;
            }
            if (gzclose(file) != Z_OK || !ok)
                throw std::runtime_error("failed writing electron creation history: " + path);
        } else {
            std::ofstream file(path, std::ios::binary);
            file.write(bytes.data(),bytes.size());
            if (!file) throw std::runtime_error("cannot write electron creation map: " + path);
        }
    }
    void Read(std::string const& path)
    {
        std::ifstream in(path, std::ios::binary);
        std::array<char,8> magic{};
        std::array<std::int32_t,2> stored_shape{};
        std::array<double,4> stored_extent{};
        std::array<double,3> info{};
        in.read(magic.data(),magic.size());
        in.read(reinterpret_cast<char*>(stored_shape.data()),sizeof(stored_shape));
        in.read(reinterpret_cast<char*>(stored_extent.data()),sizeof(stored_extent));
        in.read(reinterpret_cast<char*>(info.data()),sizeof(info));
        bool const v2 = std::string(magic.data(),8) == std::string("RREACR2\0",8);
        history_bytes = 0;
        if (v2) in.read(reinterpret_cast<char*>(&history_bytes),sizeof(history_bytes));
        if (!in || (!v2 && std::string(magic.data(),8) != std::string("RREACR1\0",8))
            || stored_shape != shape || info[0] != electron_creation_threshold_eV)
            throw std::runtime_error("missing/incompatible electron creation history; a fresh run is required");
        for (int i=0; i<4; ++i)
            if (!std::isfinite(stored_extent[i])
                || std::abs(stored_extent[i]-extent[i]) > 1e-12*(1+std::abs(extent[i])))
                throw std::runtime_error("electron creation checkpoint grid differs from the domain");
        in.read(reinterpret_cast<char*>(weights.data()),weights.size()*sizeof(double));
        if (!in || in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("truncated/oversized electron creation checkpoint");
        for (double w : weights)
            if (!std::isfinite(w) || w < 0)
                throw std::runtime_error("invalid electron creation checkpoint weight");
    }
    void RestoreHistory(std::string const& path) const
    {
        if (!std::filesystem::exists(path) && history_bytes == 0) return;
        if (std::filesystem::file_size(path) < history_bytes)
            throw std::runtime_error("electron creation history is shorter than its checkpoint");
        std::filesystem::resize_file(path,history_bytes);
    }
};
}
#endif
