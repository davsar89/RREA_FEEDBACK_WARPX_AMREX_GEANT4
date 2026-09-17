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
//   3. deflate : zlib level 1
//
// Step 2 is a pure permutation. Float values of similar magnitude share
// exponent and high-mantissa bytes, so grouping like-significance bytes exposes
// smooth-field redundancy to deflate. The compression ratio is data-dependent.
// Step 1 limits normal-value relative rounding to float32 precision; subnormal
// display values may flush to zero and must not be used as simulation state.
//
// Frames are variable length, so video_frame_metadata.csv carries the
// byte_offset and byte_length of each; that IS the index, and it is what the
// restart trim truncates on.  scripts/rrea_video_contract.py implements the
// identical codec on the read side and is pinned against this one.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <zlib.h>

#include <AMReX.H>
#include <AMReX_REAL.H>

namespace rrea::video {

inline constexpr int frame_codec_itemsize = 4;
inline constexpr int frame_codec_zlib_level = 1;

//! Narrow one array of amrex::Real into a float32 destination.
inline void NarrowToFloat32(
    amrex::Real const* src, std::size_t count, float* dst)
{
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

//! out[j*N + i] = in[i*S + j] over N elements of S bytes (blosc SHUFFLE).
inline void ByteShuffle(
    unsigned char const* in, std::size_t count, int itemsize,
    unsigned char* out)
{
    for (int j = 0; j < itemsize; ++j) {
        unsigned char* dst = out + static_cast<std::size_t>(j) * count;
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = in[i * static_cast<std::size_t>(itemsize)
                        + static_cast<std::size_t>(j)];
        }
    }
}

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
    std::vector<float> narrowed(count);
    for (std::size_t a = 0; a < arrays.size(); ++a) {
        NarrowToFloat32(
            arrays[a], values_per_array, narrowed.data() + a * values_per_array);
    }
    std::size_t const raw_bytes = count * sizeof(float);
    std::vector<unsigned char> shuffled(raw_bytes);
    ByteShuffle(
        reinterpret_cast<unsigned char const*>(narrowed.data()), count,
        frame_codec_itemsize, shuffled.data());

    uLongf bound = compressBound(static_cast<uLong>(raw_bytes));
    out.resize(static_cast<std::size_t>(bound));
    int const status = compress2(
        out.data(), &bound, shuffled.data(), static_cast<uLong>(raw_bytes),
        frame_codec_zlib_level);
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
