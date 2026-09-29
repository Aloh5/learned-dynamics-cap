"""Verify learned-model routing, derivatives, and reflection equivariance."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from ks_neumann import (NeumannKSConfig, velocity_rhs_sine, collocation_residual,
    nonlinear_sine_jacobian, spatial_wavenumbers, temporal_differentiation_matrix,
    simulate_velocity_sine)
from surrogate_neumann import fit_surrogate, QuadraticSurrogate

class SurrogateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics = fit_surrogate(8)

    def test_independent_fit_and_symmetry(self):
        self.assertLess(self.metrics['heldout_rhs_relative_l2'], 1e-12)
        self.assertEqual(self.metrics['training_periodic_orbits'], 0)
        x=np.random.default_rng(92).normal(size=(5,8))
        r=(-1.)**np.arange(1,9)
        np.testing.assert_allclose(self.model.predict(x*r),self.model.predict(x)*r,atol=1e-13)

    def test_jacobian_direction_and_adjoint(self):
        rng=np.random.default_rng(32)
        x=rng.normal(size=(7,8)); v=rng.normal(size=x.shape); w=rng.normal(size=x.shape)
        jac=self.model.jacobian(x)
        jv=np.einsum('tij,tj->ti',jac,v)
        eps=1e-5
        difference=(self.model.predict(x+eps*v)-self.model.predict(x-eps*v))/(2*eps)
        np.testing.assert_allclose(jv,difference,atol=1e-9,rtol=1e-8)
        adjoint=np.einsum('tij,ti->tj',jac,w)
        self.assertAlmostEqual(float(np.sum(w*jv)),float(np.sum(v*adjoint)),places=12)

    def test_refinement_residual_uses_learned_not_true_field(self):
        # Deliberately change the fitted vector field. A hidden true-RHS
        # residual would fail this test even when ordinary fits are exact.
        model=QuadraticSurrogate(self.model.constant+0.03,self.model.linear.copy(),self.model.quadratic.copy())
        config=NeumannKSConfig(spatial_modes=8,full_grid_nodes=32,surrogate=model)
        x=np.random.default_rng(18).normal(size=(9,8))
        expected=.13*temporal_differentiation_matrix(9)@x-model.predict(x)
        with patch('ks_neumann._spatial_operators',side_effect=AssertionError('true nonlinear evaluator called')):
            np.testing.assert_allclose(collocation_residual(x,.13,config),expected,atol=1e-13)
            np.testing.assert_allclose(velocity_rhs_sine(x,config),model.predict(x),atol=1e-13)
            k=spatial_wavenumbers(config)
            np.testing.assert_allclose(nonlinear_sine_jacobian(x[0],config),np.diag(k*k-k**4)-model.jacobian(x[0]))
            times,flow=simulate_velocity_sine(x[0]*.01,.1,config)
            self.assertTrue(np.isfinite(flow).all())
        true=collocation_residual(x,.13,replace(config,surrogate=None))
        self.assertGreater(np.linalg.norm(true-expected),.1)

if __name__=='__main__':
    unittest.main()
