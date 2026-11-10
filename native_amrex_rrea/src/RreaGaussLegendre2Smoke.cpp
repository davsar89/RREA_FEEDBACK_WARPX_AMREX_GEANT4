#include "rrea/RreaGaussLegendre2.H"
#include "rrea/RreaSmokeRequire.H"

#include <iostream>
#include <stdexcept>

int main()
{
    try {
        double const begin = 0.17;
        double const end = 0.91;
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
        auto const reversed = rrea::RreaGaussLegendre2IntegrateArray<double, 1>(
            end, begin, [](double) { return std::array<double, 1>{1.0}; });
        if (reversed[0] != 0.0) {
            throw std::runtime_error("empty/reversed interval did not return zero");
        }
    } catch (std::exception const& error) {
        std::cerr << "RREA Gauss-Legendre-2 smoke failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "RREA Gauss-Legendre-2 smoke passed\n";
    return 0;
}
