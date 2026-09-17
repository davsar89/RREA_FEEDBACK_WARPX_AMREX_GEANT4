#include "rrea/RreaGpuSmoke.H"
#include "rrea/RreaInteractionTables.H"
#include "rrea/RreaSmokeRequire.H"

#include <AMReX.H>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using rrea::smoke::require_close_rel;

}  // namespace

int main(int argc, char** argv)
{
    rrea::smoke::GpuSession gpu_runtime;
    if (argc != 2) {
        std::cerr << "usage: rrea_interaction_tables_smoke <transport_physics.json>\n";
        return EXIT_FAILURE;
    }
    try {
        // Schema-6 threshold-rate interpolation: exact zero nodes remain zero,
        // positive-positive brackets use log-log interpolation, and exact
        // inclusive endpoints are valid.
        {
            rrea::RreaInteractionTable rate;
            rate.name = "threshold_rate_smoke";
            rate.interpolation = "loglog_positive_with_zero_threshold";
            rate.extrapolation = "forbidden";
            rate.energy_eV = {1.0e3, 1.0e4, 1.0e5};
            rate.values = {0.0, 1.0, 100.0};
            // Resolves the declared interpolation and takes the node
            // logarithms; the loader runs it for every table it reads.
            rate.Finalize();
            require_close_rel(rate.Interpolate(1.0e3), 0.0, 0.0, "threshold exact zero");
            require_close_rel(rate.Interpolate(1.0e5), 100.0, 0.0, "upper exact endpoint");
            require_close_rel(
                rate.Interpolate(std::sqrt(1.0e4 * 1.0e5)),
                10.0,
                1.0e-12,
                "positive log-log midpoint");
        }
        // Conditional inverse CDFs are fail-closed: exact probability
        // endpoints, finite/nonnegative samples, and monotone sampled values.
        {
            amrex::Vector<rrea::RreaConditionalCdfGroup> valid(2);
            valid[0].primary_energy_eV = 1.0e3;
            valid[1].primary_energy_eV = 1.0e4;
            for (auto& group : valid) {
                group.cdf_u = {0.0, 0.5, 1.0};
                group.values = {1.0, 2.0, 3.0};
            }
            rrea::ValidateRreaConditionalCdfGroups(valid, "valid smoke CDF");
            auto require_rejected = [](auto groups, char const* label) {
                bool rejected = false;
                try {
                    rrea::ValidateRreaConditionalCdfGroups(groups, label);
                } catch (std::runtime_error const&) {
                    rejected = true;
                }
                if (!rejected) {
                    throw std::runtime_error(
                        std::string("conditional CDF contract accepted ") + label);
                }
            };
            auto bad = valid;
            bad[0].cdf_u.front() = 0.01;
            require_rejected(bad, "nonzero lower endpoint");
            bad = valid;
            bad[1].cdf_u.back() = 0.99;
            require_rejected(bad, "nonunit upper endpoint");
            bad = valid;
            bad[0].values[1] = -1.0;
            require_rejected(bad, "negative sampled value");
            bad = valid;
            bad[0].values[1] = std::numeric_limits<double>::quiet_NaN();
            require_rejected(bad, "nonfinite sampled value");
            bad = valid;
            bad[0].values = {1.0, 3.0, 2.0};
            require_rejected(bad, "nonmonotone sampled values");
        }
        rrea::RreaInteractionTables tables;
        tables.Load(argv[1]);
        // Load() throws on every failure path, so Loaded() only restates
        // SchemaVersion()'s one bit.
        if (tables.SchemaVersion() != 6) {
            throw std::runtime_error(
                "smoke did not load production schema 6 with a semantic configuration");
        }
        // Production Goudsmit-Saunderson elastic + positron brems load test.
        double const lam_e = tables.ElasticMeanFreePath(false, 1.0e6, 1.0);
        double const lam_p = tables.ElasticMeanFreePath(true, 1.0e6, 1.0);
        double const eta = tables.ScreeningEta(false, 1.0e6);
        if (!(lam_e > 0.0 && lam_p > 0.0 && eta > 0.0 && eta < 1.0)) {
            throw std::runtime_error("production GS elastic MFP/eta not sane");
        }
        double const c_lo = tables.SampleGsCosTheta(false, 1.0e6, 100.0, 0.01);
        double const c_hi = tables.SampleGsCosTheta(false, 1.0e6, 100.0, 0.99);
        if (!(c_hi >= c_lo && c_hi <= 1.0 && c_lo >= -1.0)) {
            throw std::runtime_error("production GS angle CDF not monotone in u");
        }
        double const c_ss = tables.SampleSingleElasticCosTheta(false, 1.0e6, 0.5);
        if (!(c_ss >= -1.0 && c_ss <= 1.0)) {
            throw std::runtime_error("production single-scatter cos out of range");
        }
        double const pk_lo = tables.SamplePositronBremsstrahlungPhotonEnergy(1.0e7, 0.1);
        double const pk_hi = tables.SamplePositronBremsstrahlungPhotonEnergy(1.0e7, 0.9);
        if (!(pk_hi >= pk_lo && pk_lo > 0.0)) {
            throw std::runtime_error("production positron brems CDF not increasing");
        }
        // Both empirical pair channels must return valid states above their
        // thresholds.  (The kinetic ledger closure is producer-guaranteed by
        // a 64-epsilon abort and deliberately not restated here.)
        {
            for (double e_gamma : {5.0e6, 2.0e7, 1.0e8}) {
                for (double u : {0.25, 0.5, 0.9}) {
                    for (bool triplet : {false, true}) {
                        auto const channel = triplet
                            ? rrea::RreaPhotonChannel::PairTriplet
                            : rrea::RreaPhotonChannel::PairNuclear;
                        auto const state = tables.SamplePairFinalState(
                            e_gamma, channel, u, 0.3, 0.7, 0.6, 0.4);
                        if (!state.valid) {
                            throw std::runtime_error("pair final state invalid above threshold");
                        }
                    }
                }
            }
        }
        // MinTableValueInRange (startup tau guard helper): a production
        // inverse-length table must be positive over its certified window.
        {
            double min_inverse_length = 0.0;
            if (!tables.MinTableValueInRange(
                    "photon_compton_inverse_length_air",
                    1.0e3,
                    1.0e11,
                    min_inverse_length)) {
                throw std::runtime_error(
                    "MinTableValueInRange failed for production Compton inverse length");
            }
            if (!(min_inverse_length > 0.0)) {
                throw std::runtime_error(
                    "production Compton inverse-length window minimum is not positive");
            }
        }
        std::cout << "rrea_interaction_tables_smoke passed\n";
    } catch (std::exception const& exc) {
        std::cerr << "rrea_interaction_tables_smoke failed: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
