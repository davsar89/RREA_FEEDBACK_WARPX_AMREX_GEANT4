#ifndef RREA_UNIFORM_STREAM_H
#define RREA_UNIFORM_STREAM_H

#include <cstdint>
#include "rrea/RreaKineticSupport.H"

namespace rrea {

// Deterministic uniform expansion for rejection samplers. Some samplers receive
// fixed pre-drawn RreaRng uniforms; a rejection loop needs an
// unbounded number.  This stream is seeded from the IEEE-754 bit
// patterns of the two pre-drawn uniforms (copied byte-wise, which is valid
// host and device code), so results are bit-reproducible for a fixed
// (seed, particle, step, subcycle) key across compilers/platforms.
//
// Policy: the FIRST proposal round of a rejection loop consumes the pre-drawn
// uniforms directly (so single-round events keep the caller's RNG draws); only
// later rounds -- and any accept draw -- pull from this stream.  With splitmix64
// mixing, an accept draw that is a deterministic hash of the proposal draws is
// statistically benign.
class RreaUniformStream {
public:
    RREA_HOST_DEVICE RreaUniformStream(double u1, double u2) {
        std::uint64_t b1 = 0;
        std::uint64_t b2 = 0;
        auto* p1=reinterpret_cast<unsigned char*>(&b1);
        auto* p2=reinterpret_cast<unsigned char*>(&b2);
        auto const* q1=reinterpret_cast<unsigned char const*>(&u1);
        auto const* q2=reinterpret_cast<unsigned char const*>(&u2);
        for (std::size_t i=0;i<sizeof(b1);++i) { p1[i]=q1[i]; p2[i]=q2[i]; }
        m_state = splitmix64(b1) ^ rotl64(splitmix64(b2), 32);
    }

    // One splitmix64 step -> a double in [0, 1).
    RREA_HOST_DEVICE double Next() {
        std::uint64_t const z = splitmix64(m_state);
        m_state += 0x9E3779B97F4A7C15ULL;
        return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
    }

private:
    RREA_HOST_DEVICE static std::uint64_t rotl64(std::uint64_t x, int r) {
        return (x << r) | (x >> (64 - r));
    }

    std::uint64_t m_state;
};

}  // namespace rrea

#endif  // RREA_UNIFORM_STREAM_H
