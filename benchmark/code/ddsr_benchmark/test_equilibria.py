"""Independent checks of polynomial derivatives and equilibrium diagnostics."""
import unittest

import numpy as np

from .equilibria import PolynomialField, dde_spectrum, search, shift_distance, stability


class EquilibriumTests(unittest.TestCase):
    def test_polynomial_jacobian_at_zero_and_nonzero_states(self):
        # f=(2+3*x^2*y, -x+4*y^3), with a zero coordinate exercising 0**0.
        class Model:
            class feature_library:
                powers_ = np.array([[0, 0], [2, 1], [1, 0], [0, 3]])

            def coefficients(self):
                return np.array([[2, 3, 0, 0], [0, 0, -1, 4]])

        field = PolynomialField(Model())
        for x, y in [(0., 0.), (0., 2.), (1.5, -.3)]:
            np.testing.assert_allclose(field(np.array([x, y])), [2+3*x*x*y, -x+4*y**3])
            np.testing.assert_allclose(field.jacobian(np.array([x, y])), [[6*x*y, 3*x*x], [-1, 12*y*y]])

    def test_dde_characteristic_equation(self):
        for c, a, unstable in [(0., .2, 1), (1., -.4, 2), (-1., -.4, 2)]:
            values, residual = dde_spectrum(c)
            self.assertLess(residual, 1e-10)
            np.testing.assert_allclose(values + .1, a*np.exp(-17*values), atol=1e-10)
            self.assertEqual(stability(values)["unstable"], unstable)
            self.assertAlmostEqual(.2*c/(1+c**10)-.1*c, 0.)

    def test_search_checks_residual_and_deduplicates(self):
        roots, attempts = search(lambda x: x*x-1, lambda x: np.diag(2*x),
                                 [np.array([2.]), np.array([3.]), np.array([-2.])])
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(roots), 2)
        np.testing.assert_allclose(sorted(x[0] for x in roots), [-1, 1])
        roots, _ = search(lambda x: np.ones_like(x), lambda x: np.zeros((1, 1)), [np.zeros(1)])
        self.assertEqual(roots, [])

    def test_discrete_shift_distance(self):
        x = np.array([1., 3., -2., 0.])
        self.assertEqual(shift_distance(x, np.roll(x, 2)), 0.)
        self.assertGreater(shift_distance(x, x+1), 0.)


if __name__ == "__main__":
    unittest.main()
