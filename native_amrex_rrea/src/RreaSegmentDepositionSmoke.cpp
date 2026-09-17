// The discrete RZ continuity identity is the test.
//
// For every fixture segment: deposit rho at the start point (CIC), fly the
// segment through DepositSegmentRZ, deposit rho at the end point into a
// second grid, and assert at EVERY cell
//
//     rho_new(c) - rho_old(c) + dt * div_idx(J)(c) == 0   (<= 1e-12 rel)
//
// plus integral checks in both index space and converted SI units.  The SI
// check uses the same annular/side areas consumed by Maxwell and asserts the
// physical finite-volume continuity equation cell by cell.
//
// Fixtures cover: in-cell hop, multi-cell diagonal, axis-touching chord,
// pure-z and pure-r moves, and a pseudo-random batch from a fixed seed.
// No AMReX dependency: the primitive is header-only plain C++, and this
// smoke must stay runnable in seconds.

#include "rrea/RreaSegmentDepositionRZ.H"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, char const* what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void run_case(char const* name, double q_w, double dt,
              double r0, double z0, double r1, double z1)
{
    rrea::SegmentGridRZ before;
    before.nr = 16; before.nz = 24;
    before.dr = 1.0; before.dz = 1.0;
    before.r_lo = 0.0; before.z_lo = -12.0;
    before.resize();
    rrea::SegmentGridRZ after = before;
    after.resize();

    rrea::DepositChargeCIC(before, q_w, r0, z0);
    rrea::DepositChargeCIC(after, q_w, r1, z1);
    rrea::DepositSegmentRZ(before, q_w, dt, r0, z0, r1, z1);

    double worst = 0.0;
    double sum_before = 0.0;
    double sum_after = 0.0;
    for (int j = 0; j < before.nz; ++j) {
        for (int i = 0; i < before.nr; ++i) {
            std::size_t const c = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(before.nr) * j;
            double const lhs = after.rho[c] - before.rho[c]
                + dt * rrea::WeightedDivergence(before, i, j);
            worst = std::fmax(worst, std::fabs(lhs));
            sum_before += before.rho[c];
            sum_after += after.rho[c];
        }
    }
    double const scale = std::fabs(q_w);
    if (worst > 1e-12 * scale) {
        std::fprintf(stderr, "FAIL %s: continuity residual %.3e (q_w %.3e)\n",
                     name, worst, q_w);
        ++failures;
    }
    check(std::fabs(sum_before - sum_after) <= 1e-12 * scale,
          "total charge conserved");

    double mz = 0.0;
    for (std::size_t f = 0; f < before.jz.size(); ++f) { mz += before.jz[f]; }
    double const mz_expected = q_w * (z1 - z0) / (before.dz * dt);
    if (std::fabs(mz - mz_expected) > 1e-12 * (std::fabs(mz_expected) + scale)) {
        std::fprintf(stderr, "FAIL %s: z moment %.6e vs expected %.6e\n",
                     name, mz, mz_expected);
        ++failures;
    }

    // Convert the exact index-space flux to the physical Maxwell face current
    // and assert the cylindrical finite-volume continuity identity in SI.
    double const q_unit_c = 1.602176634e-19;
    std::vector<double> jr_physical;
    std::vector<double> jz_physical;
    rrea::ConvertPaddedIndexCurrentToPhysicalRZ(
        before.nr, before.nz, before.dr, before.dz,
        /*pad_cells=*/0, q_unit_c,
        before.jr, before.jz, jr_physical, jz_physical);
    double physical_worst = 0.0;
    double physical_scale = 0.0;
    for (int j = 0; j < before.nz; ++j) {
        for (int i = 0; i < before.nr; ++i) {
            std::size_t const c = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(before.nr) * j;
            std::size_t const fr_lo = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(before.nr + 1) * j;
            std::size_t const fr_hi = fr_lo + 1;
            std::size_t const fz_lo = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(before.nr) * j;
            std::size_t const fz_hi = fz_lo + before.nr;
            double const r_lo_face = i * before.dr;
            double const r_hi_face = (i + 1) * before.dr;
            double const r_c = (i + 0.5) * before.dr;
            double const volume = 3.14159265358979323846
                * (r_hi_face * r_hi_face - r_lo_face * r_lo_face)
                * before.dz;
            double const delta_rho = q_unit_c
                * (after.rho[c] - before.rho[c]) / volume;
            double const div_j =
                (r_hi_face * jr_physical[fr_hi]
                 - r_lo_face * jr_physical[fr_lo])
                    / (r_c * before.dr)
                + (jz_physical[fz_hi] - jz_physical[fz_lo]) / before.dz;
            double const lhs = delta_rho + dt * div_j;
            physical_worst = std::max(physical_worst, std::abs(lhs));
            physical_scale = std::max(
                physical_scale, std::abs(delta_rho) + std::abs(dt * div_j));
        }
    }
    if (physical_worst > 2.0e-12 * std::max(physical_scale, 1.0e-300)) {
        std::fprintf(stderr,
                     "FAIL %s: physical RZ continuity residual %.3e "
                     "(scale %.3e)\n",
                     name, physical_worst, physical_scale);
        ++failures;
    }

    double physical_mz = 0.0;
    for (int j = 0; j <= before.nz; ++j) {
        for (int i = 0; i < before.nr; ++i) {
            double const r_lo_face = i * before.dr;
            double const r_hi_face = (i + 1) * before.dr;
            double const area = 3.14159265358979323846
                * (r_hi_face * r_hi_face - r_lo_face * r_lo_face);
            std::size_t const f = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(before.nr) * j;
            physical_mz += jz_physical[f] * area * before.dz;
        }
    }
    double const physical_mz_expected =
        q_unit_c * q_w * (z1 - z0) / dt;
    if (std::abs(physical_mz - physical_mz_expected)
        > 2.0e-12 * (std::abs(physical_mz_expected)
                     + std::abs(q_unit_c * q_w))) {
        std::fprintf(stderr,
                     "FAIL %s: physical z-current moment %.6e vs %.6e\n",
                     name, physical_mz, physical_mz_expected);
        ++failures;
    }
}

void radial_current_moment_interior()
{
    rrea::SegmentGridRZ g;
    g.nr = 18;
    g.nz = 12;
    g.dr = 2.0;
    g.dz = 3.0;
    g.r_lo = 0.0;
    g.z_lo = -18.0;
    g.resize();
    double const q_w = -2.3;
    double const q_unit_c = 1.602176634e-19;
    double const dt = 0.25;
    double const r0 = 4.2;
    double const r1 = 23.7;
    rrea::DepositSegmentRZ(g, q_w, dt, r0, 1.0, r1, 1.0);
    std::vector<double> jr;
    std::vector<double> jz;
    rrea::ConvertPaddedIndexCurrentToPhysicalRZ(
        g.nr, g.nz, g.dr, g.dz, 0, q_unit_c,
        g.jr, g.jz, jr, jz);
    double moment = 0.0;
    for (int j = 0; j < g.nz; ++j) {
        for (int i = 1; i <= g.nr; ++i) {
            double const r_face = i * g.dr;
            double const area = 2.0 * 3.14159265358979323846
                * r_face * g.dz;
            std::size_t const f = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(g.nr + 1) * j;
            moment += jr[f] * area * g.dr;
        }
    }
    double const expected = q_unit_c * q_w * (r1 - r0) / dt;
    check(std::abs(moment - expected)
              <= 2.0e-12 * (std::abs(expected) + std::abs(q_unit_c * q_w)),
          "physical radial-current moment matches the net chord");
}

void padded_conversion_keeps_only_physical_support()
{
    int constexpr nr = 5;
    int constexpr nz = 4;
    int constexpr pad = 2;
    int constexpr nr_pad = nr + pad;
    int constexpr nz_pad = nz + 2 * pad;
    double constexpr dr = 5.0;
    double constexpr dz = 7.0;
    double constexpr q_unit_c = 3.25;
    std::vector<double> jr_index(
        static_cast<std::size_t>(nr_pad + 1) * nz_pad, 0.0);
    std::vector<double> jz_index(
        static_cast<std::size_t>(nr_pad) * (nz_pad + 1), 0.0);
    for (int j = 0; j < nz_pad; ++j) {
        for (int i = 1; i <= nr_pad; ++i) {
            jr_index[static_cast<std::size_t>(i)
                     + static_cast<std::size_t>(nr_pad + 1) * j] =
                0.01 * (1 + 3 * i - 2 * j);
        }
    }
    for (int j = 0; j <= nz_pad; ++j) {
        for (int i = 0; i < nr_pad; ++i) {
            jz_index[static_cast<std::size_t>(i)
                     + static_cast<std::size_t>(nr_pad) * j] =
                -0.02 * (2 + i + 4 * j);
        }
    }
    std::vector<double> jr;
    std::vector<double> jz;
    rrea::ConvertPaddedIndexCurrentToPhysicalRZ(
        nr, nz, dr, dz, pad, q_unit_c,
        jr_index, jz_index, jr, jz);
    double index_radial = 0.0;
    for (int j = pad; j < pad + nz; ++j) {
        for (int i = 1; i <= nr; ++i) {
            index_radial += jr_index[static_cast<std::size_t>(i)
                + static_cast<std::size_t>(nr_pad + 1) * j];
        }
    }
    double physical_radial = 0.0;
    for (int j = 0; j < nz; ++j) {
        for (int i = 1; i <= nr; ++i) {
            double const area = 2.0 * 3.14159265358979323846
                * (i * dr) * dz;
            physical_radial += jr[static_cast<std::size_t>(i)
                                  + static_cast<std::size_t>(nr + 1) * j]
                * area;
        }
    }
    double index_axial = 0.0;
    for (int j = pad; j <= pad + nz; ++j) {
        for (int i = 0; i < nr; ++i) {
            index_axial += jz_index[static_cast<std::size_t>(i)
                + static_cast<std::size_t>(nr_pad) * j];
        }
    }
    double physical_axial = 0.0;
    for (int j = 0; j <= nz; ++j) {
        for (int i = 0; i < nr; ++i) {
            double const r_lo = i * dr;
            double const r_hi = (i + 1) * dr;
            double const area = 3.14159265358979323846
                * (r_hi * r_hi - r_lo * r_lo);
            physical_axial += jz[static_cast<std::size_t>(i)
                                 + static_cast<std::size_t>(nr) * j]
                * area;
        }
    }
    check(std::abs(physical_radial - q_unit_c * index_radial)
              <= 2.0e-12 * (std::abs(q_unit_c * index_radial) + 1.0),
          "radial conversion retains only physical transverse support");
    check(std::abs(physical_axial - q_unit_c * index_axial)
              <= 2.0e-12 * (std::abs(q_unit_c * index_axial) + 1.0),
          "axial conversion retains only physical transverse support");
}

double physical_charge_sum(
    rrea::SegmentGridRZ const& g,
    int physical_nr,
    int physical_z_begin,
    int physical_nz)
{
    double sum = 0.0;
    for (int j = physical_z_begin;
         j < physical_z_begin + physical_nz; ++j) {
        for (int i = 0; i < physical_nr; ++i) {
            sum += g.rho[static_cast<std::size_t>(i)
                         + static_cast<std::size_t>(g.nr) * j];
        }
    }
    return sum;
}

void run_escape_case(
    char const* name,
    double q_w,
    double start_r,
    double start_z,
    double micro_start_r,
    double micro_start_z,
    double crossing_r,
    double crossing_z,
    std::uint32_t faces)
{
    int constexpr physical_nr = 8;
    int constexpr physical_nz = 10;
    int constexpr pad = 2;
    double constexpr dt = 0.25;
    double constexpr q_unit = 1.0;
    rrea::SegmentGridRZ before;
    before.nr = physical_nr + pad;
    before.nz = physical_nz + 2 * pad;
    before.dr = 1.0;
    before.dz = 1.0;
    before.r_lo = 0.0;
    before.z_lo = -static_cast<double>(pad);
    before.resize();
    rrea::DepositChargeCIC(before, q_w, start_r, start_z);

    rrea::SegmentGridRZ endpoint = before;
    endpoint.resize();
    // Use the exactly snapped face, as the production primitive does.
    double snapped_r = crossing_r;
    double snapped_z = crossing_z;
    if ((faces & rrea::kEscapeFaceRHi) != 0U) {
        snapped_r = static_cast<double>(physical_nr);
    }
    if ((faces & rrea::kEscapeFaceZLo) != 0U) {
        snapped_z = 0.0;
    } else if ((faces & rrea::kEscapeFaceZHi) != 0U) {
        snapped_z = static_cast<double>(physical_nz);
    }
    rrea::DepositChargeCIC(endpoint, q_w, snapped_r, snapped_z);
    double const endpoint_valid_charge = physical_charge_sum(
        endpoint, physical_nr, pad, physical_nz);
    double const starting_valid_charge = physical_charge_sum(
        before, physical_nr, pad, physical_nz);

    rrea::SegmentGridRZ deposited = before;
    deposited.resize();
    auto const result = rrea::DepositEscapingSegmentRZ(
        deposited, q_w, dt,
        start_r, start_z,
        micro_start_r, micro_start_z,
        crossing_r, crossing_z,
        physical_nr, pad, physical_nz, faces);
    if (!result.valid) {
        std::fprintf(stderr, "FAIL %s: escape primitive rejected fixture\n", name);
        ++failures;
        return;
    }
    double const scale = std::max(std::abs(q_w), 1.0);
    check(
        std::abs(result.explicit_boundary_charge - endpoint_valid_charge)
            <= 2.0e-12 * scale,
        "explicit escape flux equals the crossing cloud's valid support");
    double jz_displacement_charge = 0.0;
    for (double const flux : deposited.jz) {
        jz_displacement_charge += dt * deposited.dz * flux;
    }
    double const expected_jz_displacement_charge =
        q_w * (snapped_z - start_z)
        + result.explicit_z_displacement_charge;
    check(
        std::abs(jz_displacement_charge - expected_jz_displacement_charge)
            <= 3.0e-12 * scale,
        "escape axial current moment includes explicit boundary-cloud sink");

    // Scratch identity remains exact on every padded cell.  On the physical
    // cells rho must be the actual disappearance of the starting CIC cloud.
    double padded_worst = 0.0;
    double physical_rho_worst = 0.0;
    for (int j = 0; j < deposited.nz; ++j) {
        for (int i = 0; i < deposited.nr; ++i) {
            std::size_t const c = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(deposited.nr) * j;
            padded_worst = std::max(
                padded_worst,
                std::abs(deposited.rho[c]
                         + dt * rrea::WeightedDivergence(deposited, i, j)));
            if (i < physical_nr && j >= pad && j < pad + physical_nz) {
                physical_rho_worst = std::max(
                    physical_rho_worst,
                    std::abs(deposited.rho[c] + before.rho[c]));
            }
        }
    }
    if (padded_worst > 3.0e-12 * scale) {
        std::fprintf(stderr,
                     "FAIL %s: padded escape continuity residual %.3e\n",
                     name, padded_worst);
        ++failures;
    }
    if (physical_rho_worst > 3.0e-12 * scale) {
        std::fprintf(stderr,
                     "FAIL %s: valid-cell rho deletion mismatch %.3e\n",
                     name, physical_rho_worst);
        ++failures;
    }

    std::vector<double> jr;
    std::vector<double> jz;
    rrea::ConvertPaddedIndexCurrentToPhysicalRZ(
        physical_nr, physical_nz, 1.0, 1.0, pad, q_unit,
        deposited.jr, deposited.jz, jr, jz);
    double physical_worst = 0.0;
    for (int j = 0; j < physical_nz; ++j) {
        for (int i = 0; i < physical_nr; ++i) {
            int const jp = j + pad;
            std::size_t const c_pad = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(deposited.nr) * jp;
            std::size_t const fr_lo = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(physical_nr + 1) * j;
            std::size_t const fr_hi = fr_lo + 1;
            std::size_t const fz_lo = static_cast<std::size_t>(i)
                + static_cast<std::size_t>(physical_nr) * j;
            std::size_t const fz_hi = fz_lo + physical_nr;
            double const r_lo = static_cast<double>(i);
            double const r_hi = static_cast<double>(i + 1);
            double const r_c = static_cast<double>(i) + 0.5;
            double const volume = 3.14159265358979323846
                * (r_hi * r_hi - r_lo * r_lo);
            double const div_j =
                (r_hi * jr[fr_hi] - r_lo * jr[fr_lo]) / r_c
                + (jz[fz_hi] - jz[fz_lo]);
            double const cell_charge_residual = deposited.rho[c_pad]
                + dt * div_j * volume / q_unit;
            physical_worst = std::max(
                physical_worst, std::abs(cell_charge_residual));
        }
    }
    if (physical_worst > 4.0e-12 * scale) {
        std::fprintf(stderr,
                     "FAIL %s: converted valid-cell continuity residual %.3e\n",
                     name, physical_worst);
        ++failures;
    }

    // Integrate signed outward current across every absorbing face.  It must
    // equal the charge which was actually supported by valid cells at the
    // beginning of the trajectory (the full macro charge for interior starts;
    // the appropriately truncated shape for an exact-face newborn).
    double outward_charge = 0.0;
    for (int j = 0; j < physical_nz; ++j) {
        std::size_t const f = static_cast<std::size_t>(physical_nr)
            + static_cast<std::size_t>(physical_nr + 1) * j;
        double const area = 2.0 * 3.14159265358979323846
            * physical_nr;
        outward_charge += dt * jr[f] * area;
    }
    for (int i = 0; i < physical_nr; ++i) {
        double const r_lo = static_cast<double>(i);
        double const r_hi = static_cast<double>(i + 1);
        double const area = 3.14159265358979323846
            * (r_hi * r_hi - r_lo * r_lo);
        outward_charge -= dt * jz[static_cast<std::size_t>(i)] * area;
        outward_charge += dt
            * jz[static_cast<std::size_t>(i)
                 + static_cast<std::size_t>(physical_nr) * physical_nz]
            * area;
    }
    if (std::abs(outward_charge - starting_valid_charge)
        > 4.0e-12 * scale) {
        std::fprintf(stderr,
                     "FAIL %s: integrated outward charge %.17g vs %.17g\n",
                     name, outward_charge, starting_valid_charge);
        ++failures;
    }
}

} // namespace

int main()
{
    double const dt = 1.0;
    run_case("in-cell hop", 3.0, dt, 4.2, 1.3, 4.7, 1.9);
    run_case("multi-cell diagonal", -2.0, dt, 2.1, -7.6, 9.8, 5.4);
    run_case("axis-touching", 5.0, dt, 0.2, 0.0, 1.9, 2.5);
    run_case("axis-crossing chord", 1.5, dt, 0.6, -1.0, 0.05, 1.0);
    run_case("pure z", 2.0, dt, 6.5, -3.0, 6.5, 8.0);
    run_case("pure r", 2.0, dt, 1.0, 4.0, 11.0, 4.0);
    run_case("zero-length", 1.0, dt, 5.5, 5.5, 5.5, 5.5);
    radial_current_moment_interior();
    padded_conversion_keeps_only_physical_support();
    // Native electrons and positrons on every absorbing plane; the crossing
    // coordinates deliberately carry tiny offsets to exercise exact snapping.
    run_escape_case("electron r_hi", -3.0,
                    4.25, 4.25, 7.25, 4.75,
                    8.0 + 2.0e-14, 4.875,
                    rrea::kEscapeFaceRHi);
    run_escape_case("positron r_hi", 2.0,
                    3.75, 5.25, 7.5, 5.0,
                    8.0 - 2.0e-14, 4.875,
                    rrea::kEscapeFaceRHi);
    run_escape_case("electron z_lo", -2.5,
                    3.25, 4.25, 3.5, 0.75,
                    3.625, -2.0e-14,
                    rrea::kEscapeFaceZLo);
    run_escape_case("positron z_hi refused upper face", 1.75,
                    4.25, 5.5, 4.5, 9.25,
                    4.625, 10.0 + 2.0e-14,
                    rrea::kEscapeFaceZHi);
    // A genuine two-face corner, a radial exit grazing z_lo (radial face only),
    // and a newborn whose residual path begins exactly on the included low-z
    // face.  Both charge signs are represented across the corner fixtures.
    run_escape_case("electron r_hi-z_hi corner", -1.25,
                    4.25, 4.25, 7.25, 9.25,
                    8.0, 10.0,
                    rrea::kEscapeFaceRHi | rrea::kEscapeFaceZHi);
    run_escape_case("positron r_hi-z_lo corner", 1.5,
                    4.5, 5.25, 7.25, 0.75,
                    8.0, 0.0,
                    rrea::kEscapeFaceRHi | rrea::kEscapeFaceZLo);
    run_escape_case("grazing radial exit on z_lo", -2.0,
                    4.25, 0.0, 7.25, 0.0,
                    8.0, 0.0,
                    rrea::kEscapeFaceRHi);
    run_escape_case("newborn on z_lo face", 2.25,
                    3.25, 0.0, 3.25, 0.0,
                    3.25, 0.0,
                    rrea::kEscapeFaceZLo);

    // Deterministic pseudo-random batch (LCG, fixed seed): the identity has
    // to hold for arbitrary chords, not curated ones.
    std::uint64_t s = 0x2b992ddfa23249d6ULL;
    auto rnd = [&s]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11) & 0x1FFFFF) / 2097152.0;
    };
    for (int k = 0; k < 200; ++k) {
        // Interior in z so this random batch isolates ordinary chords; the
        // explicit escape fixtures above own absorbing-edge closure.  Keep it
        // deliberately axis-grazing in r.
        double const r0 = 0.05 + 12.0 * rnd();
        double const z0 = -9.0 + 18.0 * rnd();
        run_case("random", (rnd() < 0.5 ? -1.0 : 1.0) * (0.5 + rnd()), dt,
                 r0, z0,
                 std::fmax(0.01, r0 + 4.0 * (rnd() - 0.5)),
                 z0 + 4.0 * (rnd() - 0.5));
    }

    if (failures) {
        std::fprintf(stderr, "rrea_segment_deposition_smoke: %d FAILURES\n",
                     failures);
        return 1;
    }
    std::printf("rrea_segment_deposition_smoke: PASS (215 cases, index/SI "
                "continuity, exact charged-boundary closure, current moments, "
                "and absorbing-pad loss exact)\n");
    return 0;
}
