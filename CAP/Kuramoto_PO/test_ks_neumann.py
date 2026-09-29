"""Unit tests for the Neumann/odd KS formulation (stdlib unittest)."""

import unittest

import numpy as np

from diagnose_smooth_cap import (
    ahat_column,
    direct_ahat_column,
    direct_raw_tail_column,
    tail_preconditioned_nonlinear_column,
)

from ks_neumann import (
    NeumannKSConfig,
    boundary_defects,
    collocation_residual,
    nonlinear_sine_coefficients,
    nonlinear_sine_jacobian,
    potential_coefficients_from_velocity,
    temporal_differentiation_matrix,
    time_fourier_coefficients,
    time_values_from_fourier,
    velocity_coefficients_from_potential,
)


class NeumannKSTests(unittest.TestCase):
    def setUp(self):
        self.config = NeumannKSConfig(
            spatial_modes=8, full_grid_nodes=64, temporal_modes=4
        )

    def test_boundary_conditions_are_built_into_basis(self):
        rng = np.random.default_rng(3)
        coefficients = rng.standard_normal((5, self.config.spatial_modes))
        defects = boundary_defects(coefficients, self.config)
        self.assertLess(defects["velocity_value"], 1e-12)
        self.assertLess(defects["velocity_second_derivative"], 1e-11)
        self.assertLess(defects["potential_first_derivative"], 1e-12)
        self.assertLess(defects["potential_third_derivative"], 1e-11)

    def test_potential_velocity_round_trip(self):
        rng = np.random.default_rng(4)
        velocity = rng.standard_normal(self.config.spatial_modes)
        potential = potential_coefficients_from_velocity(velocity, self.config)
        recovered = velocity_coefficients_from_potential(potential, self.config)
        np.testing.assert_allclose(recovered, velocity, atol=1e-14, rtol=1e-14)

    def test_temporal_derivative(self):
        node_count = 9
        theta = 2.0 * np.pi * np.arange(node_count) / node_count
        values = np.sin(3.0 * theta)
        derivative = temporal_differentiation_matrix(node_count) @ values
        np.testing.assert_allclose(derivative, 3.0 * np.cos(3.0 * theta), atol=1e-12)

    def test_nonlinear_jacobian(self):
        rng = np.random.default_rng(5)
        state = 0.2 * rng.standard_normal(self.config.spatial_modes)
        direction = rng.standard_normal(self.config.spatial_modes)
        epsilon = 1e-7
        difference = (
            nonlinear_sine_coefficients(state + epsilon * direction, self.config)
            - nonlinear_sine_coefficients(state - epsilon * direction, self.config)
        ) / (2.0 * epsilon)
        analytic = nonlinear_sine_jacobian(state, self.config) @ direction
        np.testing.assert_allclose(difference, analytic, atol=2e-9, rtol=2e-8)

    def test_time_fourier_round_trip(self):
        rng = np.random.default_rng(6)
        values = rng.standard_normal((9, self.config.spatial_modes))
        recovered = time_values_from_fourier(time_fourier_coefficients(values))
        np.testing.assert_allclose(recovered, values, atol=1e-13, rtol=1e-13)

    def test_constant_zero_is_exact_equilibrium(self):
        values = np.zeros((9, self.config.spatial_modes))
        residual = collocation_residual(values, 1.0, self.config)
        self.assertEqual(float(np.max(np.abs(residual))), 0.0)

    def test_generic_translation_leaves_fixed_boundary_space(self):
        # u(x)=sin(pi*x/ell) satisfies u(0)=u(ell)=0.  A generic translate
        # has u_delta(0)=sin(pi*delta/ell) != 0, so translations do not act
        # continuously within the boundary-value phase space.
        delta = 0.37
        shifted_endpoint = np.sin(np.pi * delta / self.config.interval_length)
        self.assertGreater(abs(shifted_endpoint), 1e-3)

    def test_direct_fourier_column_matches_matrix_construction(self):
        rng = np.random.default_rng(7)
        centered = rng.standard_normal((9, 5)) + 1j * rng.standard_normal((9, 5))
        centered[4] = centered[4].real
        centered[:4] = np.conj(centered[:4:-1])
        input_modes = 7
        harmonic = 3
        input_basis = input_modes + harmonic
        old = ahat_column(centered, input_modes, input_basis, 2, 3.2, 0.08)
        direct = direct_ahat_column(centered, harmonic, True, 2, 3.2, 0.08)
        old_modes = (len(old) - 1) // 2
        direct_modes = (len(direct) - 1) // 2
        restricted = np.r_[
            old[0].ravel(),
            old[1 : direct_modes + 1].ravel(),
            old[old_modes + 1 : old_modes + direct_modes + 1].ravel(),
        ]
        np.testing.assert_allclose(restricted, direct.ravel(), atol=2e-14, rtol=2e-14)

        old_raw = tail_preconditioned_nonlinear_column(
            centered, input_modes, input_basis, 2, 3.2, 0.08
        )
        direct_raw = direct_raw_tail_column(
            centered, harmonic, True, 2, 3.2, 0.08
        )
        restricted_raw = np.r_[
            old_raw[0].ravel(),
            old_raw[1 : direct_modes + 1].ravel(),
            old_raw[old_modes + 1 : old_modes + direct_modes + 1].ravel(),
        ]
        np.testing.assert_allclose(
            restricted_raw, direct_raw.ravel(), atol=2e-14, rtol=2e-14
        )


if __name__ == "__main__":
    unittest.main()
