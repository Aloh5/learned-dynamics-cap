"""Regression invariants, scaling conventions, export and exact derivatives."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from dpsr.polynomial import (PolynomialLibrary, PolynomialModel,
    polynomial_exponents, fit_polynomial_vector_field, solve_dense_regression)
from dpsr.dynamics import cumulative_error_horizon, nrmse


class PolynomialTests(unittest.TestCase):
    def test_complete_library_and_exact_recovery(self):
        x = np.random.default_rng(3).normal(size=(200, 3))
        powers = polynomial_exponents(3, 2)
        self.assertEqual(powers.shape, (10, 3))
        coefficients = np.random.default_rng(4).normal(size=(3, 10))
        y = PolynomialLibrary(powers).transform(x) @ coefficients.T
        model = fit_polynomial_vector_field(x, y)
        np.testing.assert_allclose(model.coefficients(), coefficients, atol=1e-13)
        self.assertTrue(model.diagnostics['full_column_rank'])
        np.testing.assert_allclose(model.predict(x), y, atol=1e-13)

    def test_rank_deficient_fit_preserves_unscaled_minimum_norm(self):
        x = np.random.default_rng(8).normal(size=(80, 3))
        x[:, 2] = 2-x[:, 0]-x[:, 1]
        theta = PolynomialLibrary(polynomial_exponents(3, 2)).transform(x)
        y = np.column_stack([x[:, 0]**2, x[:, 1], x[:, 2]])
        model = fit_polynomial_vector_field(x, y)
        expected = np.linalg.lstsq(theta, y, rcond=np.finfo(float).eps)[0]
        self.assertFalse(model.diagnostics['full_column_rank'])
        np.testing.assert_allclose(model.coefficients().T, expected, atol=1e-12)
        np.testing.assert_allclose(model.predict(x), y, atol=1e-12)

    def test_normalized_degree_weighted_ridge_objective(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(150, 2))*[.02, 30]
        y = rng.normal(size=(150, 2))
        model = fit_polynomial_vector_field(x, y, regularization=.07, normalize=True)
        powers = model.feature_library.powers_
        theta = model.feature_library.transform(x)
        scales = np.sqrt(np.mean(theta**2, axis=0))
        normalized = theta/scales
        weights = np.maximum(powers.sum(axis=1), 1).astype(float)
        weights[0] = 0
        fitted = model.coefficients()*scales
        gradient = normalized.T @ (normalized @ fitted.T-y) + .07*weights[:, None]*fitted.T
        np.testing.assert_allclose(gradient, 0, atol=1e-11)

    def test_analytic_jacobian_and_roundtrip_at_zero_coordinates(self):
        powers = np.array([[0, 0], [2, 1], [0, 3]])
        model = PolynomialModel([[2, 3, 0], [0, 0, 4]], powers)
        for x in [np.array([0., 2.]), np.array([1.5, -.3])]:
            expected = [[6*x[0]*x[1], 3*x[0]**2], [0, 12*x[1]**2]]
            np.testing.assert_allclose(model.jacobian(x), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.npz'
            model.save(path)
            restored = PolynomialModel.load(path)
            np.testing.assert_array_equal(restored.predict([[0, 2], [1.5, -.3]]),
                                          model.predict([[0, 2], [1.5, -.3]]))

    def test_metric_coordinate_conventions(self):
        truth = np.array([[1., 3.], [2., 4.], [3., 5.]])
        pred = truth + [1., 0.]
        self.assertEqual(cumulative_error_horizon(truth, pred, 1, .6, coordinate=0), 0)
        self.assertEqual(cumulative_error_horizon(truth, pred, 1, .6), 1)
        self.assertTrue(np.isnan(nrmse(truth, None)))

    def test_reject_invalid_training_data(self):
        with self.assertRaises(ValueError):
            solve_dense_regression([[np.nan]], [[0]])
        with self.assertRaises(ValueError):
            fit_polynomial_vector_field(np.ones((3, 2)), np.ones((3, 1)))


if __name__ == '__main__':
    unittest.main()
