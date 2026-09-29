"""Focused numerical/protocol checks; not tests of the CAP certificates."""
import unittest
import numpy as np
from .paper_comparison import Predictor, score, histories_and_truth
from .models import fit_ddsr, fit_sindy_stlsq
from .dynamics import ks_rhs_batch


class ComparisonChecks(unittest.TestCase):
    def test_complete_quadratic_recovers_spectral_rhs(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=(500, 14))
        x -= x.mean(axis=1, keepdims=True)
        target = ks_rhs_batch(x)
        model = fit_ddsr(x, target, 2, 0)
        heldout = rng.normal(size=(50, 14)); heldout -= heldout.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(model.predict(heldout), ks_rhs_batch(heldout), atol=1e-10, rtol=1e-10)

    def test_threshold_zero_matches_unregularized_dense(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(150, 3)); y = rng.normal(size=(150, 3))
        dense = fit_ddsr(x, y, 2, 0)
        sparse = fit_sindy_stlsq(x, y, 2, 0, 0)
        np.testing.assert_allclose(dense.predict(x), sparse.predict(x), atol=1e-12)

    def test_rk4_batch_matches_linear_flow_and_refines(self):
        class Linear:
            def predict(self, x):
                return -x
        p = Predictor.__new__(Predictor)
        p.kind='DPSR'; p.dt=.2; p.dimension=1
        p.mean=np.zeros(1); p.scale=np.ones(1); p.model=Linear()
        history=np.array([[[1.],[1.]], [[2.],[2.]]])
        a=p.rollout(history,5,substeps=1)[:,-1,0]
        b=p.rollout(history,5,substeps=2)[:,-1,0]
        exact=np.exp(-1)*np.array([1.,2.])
        self.assertLess(np.linalg.norm(b-exact),np.linalg.norm(a-exact)/10)

    def test_failed_rollouts_stay_in_denominator(self):
        truth=np.zeros((2,4,1)); pred=truth.copy(); pred[1,2:]=np.nan
        rows=score(dict(scalar=True,dt=.1,tolerance=.05),pred,truth,1)
        self.assertTrue(rows[0]['completed']); self.assertFalse(rows[1]['completed'])
        self.assertEqual(rows[1]['horizon'],0)
        self.assertTrue(np.isnan(rows[1]['nrmse']))

    def test_initialization_contains_no_future_truth(self):
        values=np.arange(300)[:,None]
        history,truth,starts=histories_and_truth(dict(dt=.1),values,5,3)
        np.testing.assert_array_equal(history[:,-1],truth[:,0])
        np.testing.assert_array_equal(history[:,0,0],starts-50)
        np.testing.assert_array_equal(truth[:,-1,0],starts+50)


if __name__ == '__main__':
    unittest.main()
