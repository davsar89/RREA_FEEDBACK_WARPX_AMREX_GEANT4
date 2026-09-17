#include "rrea/RreaGaussLegendre8.H"
#include "rrea/RreaSmokeRequire.H"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double antiderivative(double x)
{
    // A deliberately nonuniform degree-14 field profile.  An eight-point
    // Gauss-Legendre rule integrates it exactly apart from roundoff.
    double sum = 0.0;
    double power = x;
    for (int degree = 0; degree <= 14; ++degree) {
        double const coefficient = (degree % 2 == 0 ? 1.0 : -0.375)
            * static_cast<double>(degree + 1);
        sum += coefficient * power / static_cast<double>(degree + 1);
        power *= x;
    }
    return sum;
}

double profile(double x)
{
    double sum = 0.0;
    double power = 1.0;
    for (int degree = 0; degree <= 14; ++degree) {
        double const coefficient = (degree % 2 == 0 ? 1.0 : -0.375)
            * static_cast<double>(degree + 1);
        sum += coefficient * power;
        power *= x;
    }
    return sum;
}


}  // namespace

int main()
{
    try {
        double const begin = 0.17;
        double const split = 0.43;
        double const end = 0.91;
        std::vector<double> samples;
        double const whole = rrea::RreaGaussLegendre8Integrate<double>(
            begin,
            end,
            [&](double x) {
                samples.push_back(x);
                return profile(x);
            });
        double const expected = antiderivative(end) - antiderivative(begin);
        rrea::smoke::require_close(whole, expected, "degree-14 nonuniform profile", 256.0);
        if (samples.size() != 8U) {
            throw std::runtime_error("Gauss-Legendre helper did not use exactly eight samples");
        }
        for (std::size_t i = 0; i < samples.size(); i += 2U) {
            if (!(samples[i] < samples[i + 1U])) {
                throw std::runtime_error("Gauss-Legendre sample order is not fixed -,+");
            }
        }

        double const pieces = rrea::RreaGaussLegendre8Integrate<double>(
            begin, split, profile)
            + rrea::RreaGaussLegendre8Integrate<double>(split, end, profile);
        rrea::smoke::require_close(pieces, expected, "piecewise clipped profile", 256.0);
        if (rrea::RreaGaussLegendre8Integrate<double>(end, begin, profile) != 0.0) {
            throw std::runtime_error("empty/reversed interval did not return zero");
        }

        // The two-node rule is exact for cubics: that is what makes it the
        // right quadrature for the field work inside one cell, where the
        // bilinear gather along an affine chord is quadratic.
        auto const cubic = rrea::RreaGaussLegendre2IntegrateArray<double, 2>(
            begin, end, [](double x) {
                return std::array<double, 2>{
                    2.0 - 3.0 * x + 5.0 * x * x - 1.5 * x * x * x, 1.0};
            });
        auto const cubic_antiderivative = [](double x) {
            return 2.0 * x - 1.5 * x * x + (5.0 / 3.0) * x * x * x
                - 0.375 * x * x * x * x;
        };
        rrea::smoke::require_close(
            cubic[0],
            cubic_antiderivative(end) - cubic_antiderivative(begin),
            "two-node cubic exactness", 16.0);
        rrea::smoke::require_close(
            cubic[1], end - begin, "two-node constant component", 16.0);
    } catch (std::exception const& error) {
        std::cerr << "RREA Gauss-Legendre-8 smoke failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "RREA Gauss-Legendre-8 smoke passed\n";
    return 0;
}
