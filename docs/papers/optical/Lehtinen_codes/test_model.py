"""Domain and failure-path checks; long preset convergence is a separate command."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
import scipy.constants as phys
import fgf_module as f
import bethe
from egu2025poster import configure_case, get_results
from optical_xray import solve_runaway_transport, nu1NN2p_fun


class ModelTests(unittest.TestCase):
    def setUp(self):
        configure_case('FGF')
        f.init()

    def solution(self, **kwargs):
        t = np.array([f.T1, f.T2])
        values = dict(success=True, status=0, message='test', t=t,
                      y=np.array([[f.E0]*2, [f.NR_cloud]*2, [f.Nion_cloud]*2]),
                      sol=None, fgf_t_span=(f.T1, f.T2), fgf_atol=f.DEFAULT_ATOL)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_source_activation_follows_amplitudes(self):
        f.KEext, f.KJext = 0., 2.
        f.init()
        self.assertFalse(f.external_field)
        self.assertTrue(f.external_current)
        u = [f.E0, f.NR_cloud, f.Nion_cloud]
        conduction = f.state_quantities(0., u)['Jtot']
        assert_allclose(f.fgf(0., u)[0]*phys.epsilon_0, f.J0+f.Jext(0.)-conduction)
        f.KEext, f.KJext = -1., 0.
        f.init()
        self.assertTrue(f.external_field)
        self.assertFalse(f.external_current)

    def test_currents_obey_field_equation_all_branches(self):
        f.KEext = 0
        f.init()
        for use_ions, minimal, instantaneous in [(True, False, False), (False, False, False),
                                                (True, True, True), (True, False, True)]:
            f.use_ions, f.minimal_model = use_ions, minimal
            f.do_feedback, f.instantaneous_NR = not instantaneous, instantaneous
            for E in [-f.Et, 0., .01, f.Et]:
                with self.subTest(ions=use_ions, minimal=minimal, instantaneous=instantaneous, E=E):
                    u = np.array([E, 1., f.Nion_cloud])
                    out = f.sol2values(self.solution(y=np.repeat(u[:, None], 2, axis=1)))
                    expected = f.J0-out['Jtot'][0]
                    assert_allclose(f.fgf(0., u)[0]*phys.epsilon_0, expected, atol=1e-25)
                    self.assertTrue(all(np.all(np.isfinite(x)) for x in out.values()))
                    if E < 0:
                        self.assertLess(out['JR'][0], 0)
                    if not use_ions:
                        assert_allclose(out['Jion'][0], np.copysign(f.sigma_cloud*max(abs(E), .1), E))
                    if minimal:
                        assert_allclose(out['Jtot'], out['Jion'])
                    else:
                        assert_allclose(out['Jtot'], out['JR']+out['Jsec']+out['Jion'])

    def test_radio_inverse_distance_and_sampling(self):
        t = np.linspace(-.001, .001, 1001)
        current = np.exp(-(t/.0001)**2)*1e-6
        a = f.get_Erad(t, current, 10000)
        b = f.get_Erad(t, current, 20000)
        assert_allclose(a, 2*np.array(b), atol=0)
        for distance in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                f.get_Erad(t, current, distance)
        with self.assertRaises(ValueError):
            f.get_Erad(t[::-1], current)
        with self.assertRaises(ValueError):
            f.get_Erad(np.array([0., 1., 3.]), np.zeros(3))
        # For a linear current moment, central differentiation gives its exact slope.
        result, _ = f.get_Erad(t, 2*t, 10000)
        assert_allclose(result[1:-1], 1e-7*2*f.Lcloud**3/10000, rtol=1e-12)

    def test_failed_incomplete_and_invalid_results_rejected(self):
        for sol in [self.solution(success=False, status=-1), self.solution(status=1),
                    self.solution(t=np.array([f.T1, 0.])),
                    self.solution(y=np.full((3, 2), np.nan))]:
            with self.assertRaises(RuntimeError):
                f.sol2values(sol)
        sol = self.solution()
        sol.y[1, 0] = -1e-10
        with self.assertRaises(RuntimeError):
            f.sol2values(sol)
        sol.y[1, 0] = -1e-14
        with self.assertWarns(RuntimeWarning):
            self.assertLess(f.sol2values(sol)['NR'][0], 0)

    def test_genuine_solver_failure_does_not_extrapolate(self):
        f.T1, f.T2, f.E0, f.num_time_points = 0., 2., 1., 101
        with patch.object(f, 'fgf', lambda t, u: [u[0]**2, 0., 0.]):
            with self.assertRaisesRegex(RuntimeError, 'Integration incomplete'):
                f.solve_odes()

    def test_explicit_resampling_and_returned_grid(self):
        sol = self.solution()
        f.num_time_points = 999
        self.assertEqual(len(f.sol2values(sol)['t']), 2)
        sol.sol = lambda t: np.repeat(sol.y[:, :1], len(t), axis=1)
        self.assertEqual(len(f.sol2values(sol, t=np.linspace(f.T1, f.T2, 5))['t']), 5)
        with self.assertRaises(ValueError):
            f.sol2values(sol, t=[f.T1, f.T2+1])

    def test_legacy_volume_threshold_and_warning(self):
        with np.errstate(all='raise'):
            self.assertEqual(f.get_volume(float(f.Et)), f.Lcloud**3)
            assert_allclose(f.get_NRext(f.Et), f.S*f.Lcloud/phys.c)
            with self.assertWarnsRegex(RuntimeWarning, 'unresolved'):
                volume = f.get_volume(np.array([-f.Et, f.Et-1, f.Et, f.Et+1]))
            assert_allclose(volume, [0, 0, f.Lcloud**3, f.Lcloud**3])

    def test_optical_source_species_normalization(self):
        # Explicit sum of ionization collision rates for N2 and O2.
        Nm = f.N0*f.Katm
        NR, E = 2., f.Et+1000.
        source = phys.c*NR*10*(.8*Nm*f.sigma_i_general(f.UR,16,14,13)
                                  +.2*Nm*f.sigma_i_general(f.UR,16,16,17.4))
        supra = f.electron_thermalization_time(16)*source
        yield_probability = 2.38e7/(2.38e7+Nm*(.2*3e-16+.8*.13e-16))
        expected = supra*(.8*Nm)*3.8e-21*f.vfun(16)*yield_probability*f.get_volume(E)
        assert_allclose(f.get_337(E, NR), expected, rtol=1e-14)
        NRarray = np.array([1., 3., 2.])
        Earray = f.Et+np.array([100., 10000., 50000.])
        ratio = f.get_337(Earray, NRarray)/f.get_Fgrays(NRarray, Earray)
        assert_allclose(ratio, ratio[0], rtol=1e-14)
        self.assertEqual(f.get_Fgrays(1., E, minTGFeV=f.UR), 0.)

    def test_preset_resets_and_unknown_case(self):
        with patch.object(f, 'init_and_solve', side_effect=lambda **kw: (f.ode_method, f.Lcloud, f.minimal_model)):
            baseline = get_results('FGF')
            get_results('TGF2')
            f.minimal_model = True
            self.assertEqual(get_results('FGF'), baseline)
            with self.assertRaises(ValueError):
                get_results('typo')

    def test_optional_feedback_guards_and_manual_similarity(self):
        f.constant_delta = False
        with self.assertRaises(NotImplementedError):
            f.init()
        f.constant_delta, f.Katm = True, 1.2
        with self.assertRaises(NotImplementedError):
            f.init()
        f.manual_feedback_parameters = True
        f.init()
        assert_allclose(f.Ef/f.Katm, 2.8e5+f.UR/200)
        assert_allclose(f.A/f.Katm, 1/200)


class InteractionTests(unittest.TestCase):
    def test_small_field_langevin_limit(self):
        with np.errstate(all='raise'):
            self.assertEqual(bethe.mu_ave(0.), 0.)
            assert_allclose(bethe.mu_ave(np.array([1e-12, 1e-8]))/np.array([1e-12, 1e-8]), 2/3)
            assert_allclose(bethe.mu_ave(1e6), 1., rtol=1e-6)
            self.assertEqual(bethe.vdrift(bethe.pmc_from_keV(1000.), 0.), 0.)

    def test_brems_broadcast_and_kinematic_mask(self):
        p = np.array([[0.], [1.], [10.]])
        k = np.array([.01, .5, 10.])
        with np.errstate(all='raise'):
            cross = bethe.bremsstrahlung_3BN(p, k)
            self.assertEqual(cross.shape, (3, 3))
            self.assertTrue(np.all(cross >= 0))
            for i, momentum in enumerate(p[:, 0]):
                for j, photon in enumerate(k):
                    assert_allclose(cross[i, j], bethe.bremsstrahlung_3BN(momentum, photon))
            self.assertEqual(bethe.bremsstrahlung_3BN(1., np.sqrt(2)-1), 0.)
        # Values well inside the valid domain, verified against the original formula.
        assert_allclose(cross[1, 0], 3.55285541e-26, rtol=2e-9)

    def test_runaway_cross_section_broadcast(self):
        energies = np.array([0., 1e3, 1e6, 1e7])*phys.e
        fields = np.array([1e5, 5e5, 1e6])
        with np.errstate(all='raise'):
            matrix = bethe.sigma_re(energies[:, None], fields)
            for i, energy in enumerate(energies):
                for j, field in enumerate(fields):
                    assert_allclose(matrix[i, j], bethe.sigma_re(energy, field))
            self.assertTrue(np.all(matrix[:2] == 0))
            # Infinite primary energy is a documented, finite physical limit.
            expected = 2*np.pi*bethe.Zm*bethe.r0**2*bethe.mc2/bethe.get_En_th(1e6)
            assert_allclose(bethe.sigma_re(np.inf, 1e6), expected)


class StreamerTransportTests(unittest.TestCase):
    def test_localized_source_is_not_skipped(self):
        x = np.linspace(0., 1., 1001)
        source = np.zeros_like(x)
        source[500] = 1.
        for velocity in (1., -1.):
            density = solve_runaway_transport(x, 0., source, velocity)
            outflow = density[-1] if velocity > 0 else density[0]
            assert_allclose(outflow*abs(velocity), np.trapezoid(source, x), rtol=1e-7)

    def test_constant_coefficient_and_both_flow_directions(self):
        x = np.linspace(-1., 2., 101)
        for velocity in (2., -2.):
            age = (x-x[0])/velocity if velocity > 0 else (x[-1]-x)/abs(velocity)
            for growth in (0., .2, -.2):
                expected = 3*age if growth == 0 else 3*np.expm1(growth*age)/growth
                assert_allclose(solve_runaway_transport(x, growth, 3., velocity), expected, rtol=2e-9, atol=1e-11)
        with self.assertRaises(ValueError):
            solve_runaway_transport(x, 1., 1., 0.)

    def test_varying_coefficient_integrating_factor(self):
        x = np.linspace(0., 1., 101)
        A = lambda z: .2*z+z*z/2
        expected = np.exp(A(1))*quad(lambda z: np.exp(-A(z)), 0., 1.)[0]
        actual = solve_runaway_transport(x, .2+x, 1., 1.)
        assert_allclose(actual[-1], expected, rtol=1e-9)
        self.assertAlmostEqual(expected, 1.574111, places=5)
        with self.assertRaises(NotImplementedError):
            nu1NN2p_fun(1e6)


if __name__ == '__main__':
    unittest.main()
