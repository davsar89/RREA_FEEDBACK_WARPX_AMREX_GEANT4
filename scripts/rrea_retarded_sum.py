"""Compiled time interpolation for the recorded Love-equivalence surface.

Each kernel contributes c0*f(t-R/c) + c1*df/dt, with linear source-time
interpolation, zero prehistory, and the supplied separate E/B time origins.
Geometry and quadrature coefficients belong to compute_rrea_em_radio.py.
Pins: analytic Coulomb/dipole, causal fronts and axis tests in test_radio_physics.
"""

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def retarded_surface_sum(window, times, origin, source_count, dt,
                         offsets, delays, source_t0, c_value, c_derivative):
    """Sum one owned time window; independent observer times run in parallel."""
    n_components = c_value.shape[1]
    value_out = np.zeros((len(times), n_components))
    derivative_out = np.zeros_like(value_out)
    for i in prange(len(times)):
        t = times[i]
        for j in range(len(offsets)):
            if t < source_t0[j] + delays[j]:
                continue
            x = min((t - delays[j] - source_t0[j]) / dt, source_count - 1.)
            k = max(0, min(int(math.floor(x)), source_count - 2))
            fraction = max(0., min(x - k, 1.))
            f0 = window[k - origin, offsets[j]]
            difference = window[k - origin + 1, offsets[j]] - f0
            value = f0 + fraction * difference
            derivative = difference / dt
            for c in range(n_components):
                value_out[i, c] += c_value[j, c] * value
                derivative_out[i, c] += c_derivative[j, c] * derivative
    return value_out, derivative_out
