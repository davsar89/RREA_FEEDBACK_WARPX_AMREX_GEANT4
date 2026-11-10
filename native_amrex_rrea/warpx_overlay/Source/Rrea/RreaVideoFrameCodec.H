#ifndef RREA_VIDEO_FRAME_CODEC_H
#define RREA_VIDEO_FRAME_CODEC_H

// Encoding for the rrea_video_frames_binary_v4 frame stream.
//
// A frame is five display-resampled nz*nr scalar fields.  Each is narrowed to
// float32 and given one lossless permutation plus deflate, per frame,
// independently:
//
//   1. narrow  : float32(x)                   -- the ONLY lossy step
//   2. shuffle : out[j*N + i] = in[i*S + j]    -- blosc SHUFFLE, S = 4 bytes
//   3. deflate : zlib level 6 (lossless)
//
// Step 2 is a pure permutation. Float values of similar magnitude share
// exponent and high-mantissa bytes, so grouping like-significance bytes exposes
// smooth-field redundancy to deflate. The compression ratio is data-dependent.
// Step 1 limits normal-value relative rounding to float32 precision; subnormal
// display values may flush to zero and must not be used as simulation state.
//
// Frames are variable length, so video_frame_metadata.csv carries the
// byte_offset and byte_length of each; that IS the index, and it is what the
// restart trim truncates on.  scripts/rrea_video_stream.py decodes it
// (inflate, then rrea_table_pack.byte_unshuffle).

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include <zlib.h>

#include <AMReX.H>
#include <AMReX_REAL.H>

namespace rrea::video {

/*! Encode one whole frame: narrow, shuffle, deflate.
 *
 * \param arrays  the five field buffers, in the fixed v4 array order
 * \param out     receives the compressed blob (resized to the exact length)
 *
 * Aborts rather than writing a short or unverifiable frame: a truncated frame
 * would desynchronize every later byte_offset in the stream.
 */
inline void EncodeFrame(
    std::vector<amrex::Real const*> const& arrays, std::size_t values_per_array,
    std::vector<unsigned char>& out)
{
    std::size_t const count = arrays.size() * values_per_array;
    std::vector<unsigned char> shuffled(count * sizeof(float));
    for (std::size_t i = 0; i < count; ++i) {
        float const value = static_cast<float>(
            arrays[i / values_per_array][i % values_per_array]);
        unsigned char bytes[sizeof(float)];
        std::memcpy(bytes, &value, sizeof(float));
        for (std::size_t j = 0; j < sizeof(float); ++j) {
            shuffled[j * count + i] = bytes[j];
        }
    }
    uLongf bound = compressBound(static_cast<uLong>(shuffled.size()));
    out.resize(static_cast<std::size_t>(bound));
    int const status = compress2(
        out.data(), &bound, shuffled.data(), static_cast<uLong>(shuffled.size()),
        /*level=*/6);
    if (status != Z_OK) {
        amrex::Abort(
            "rrea video-v4: zlib compress2 failed with status "
            + std::to_string(status)
            + " -- refusing to append an unencodable frame");
    }
    out.resize(static_cast<std::size_t>(bound));
}

}  // namespace rrea::video

#endif  // RREA_VIDEO_FRAME_CODEC_H
