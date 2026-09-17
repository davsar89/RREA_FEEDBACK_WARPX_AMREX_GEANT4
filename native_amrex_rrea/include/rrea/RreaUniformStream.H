#ifndef RREA_UNIFORM_STREAM_H
#define RREA_UNIFORM_STREAM_H

#include <cstdint>
#include <cstring>

namespace rrea {

// Deterministic uniform expansion for rejection samplers. Some samplers receive
// fixed pre-drawn RreaRng uniforms; a rejection loop needs an
// unbounded number.  This stream is seeded from the IEEE-754 bit
// patterns of the two pre-drawn uniforms (via std::memcpy -- portable, no
// std::bit_cast), so results are bit-reproducible for a fixed
// (seed, particle, step, subcycle) key across compilers/platforms.
//
// Policy: the FIRST proposal round of a rejection loop consumes the pre-drawn
// uniforms directly (so single-round events keep the caller's RNG draws); only
// later rounds -- and any accept draw -- pull from this stream.  With splitmix64
// mixing, an accept draw that is a deterministic hash of the proposal draws is
// statistically benign (the Tier-2 CDF-gap comparator is the guard).
class RreaUniformStream {
public:
    RreaUniformStream(double u1, double u2) {
        std::uint64_t b1 = 0;
        std::uint64_t b2 = 0;
        std::memcpy(&b1, &u1, sizeof(b1));
        std::memcpy(&b2, &u2, sizeof(b2));
        m_state = splitmix64(b1) ^ rotl64(splitmix64(b2), 32);
    }

    // One splitmix64 step -> a double in [0, 1).
    double Next() {
        m_state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = m_state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
    }

private:
    static std::uint64_t splitmix64(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }
    static std::uint64_t rotl64(std::uint64_t x, int r) {
        return (x << r) | (x >> (64 - r));
    }

    std::uint64_t m_state;
};

}  // namespace rrea

#endif  // RREA_UNIFORM_STREAM_H
