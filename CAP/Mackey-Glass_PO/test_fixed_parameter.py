"""Numerical cross-checks and certificate consistency (not interval proofs)."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.io import loadmat
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate',HERE/'fixed_parameter_candidate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FixedParameterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=loadmat(HERE/'outputs_fixed_parameter/mg_fixed_parameter_candidate.mat')
    def test_frequency_and_state_jacobian_against_difference(self):
        d=self.d;u=d['u'].ravel();omega=d['omega'].item();ref=d['reference_omega'].item();phase=d['phase'].ravel()
        rng=np.random.default_rng(12);h=rng.normal(size=len(u))+1j*rng.normal(size=len(u));h=(h+h[::-1].conj())/2;h/=np.linalg.norm(h,1)
        dw=.31;f,J=m.system(omega,u,ref,phase)
        eps=1e-6
        fp,_=m.system(omega+eps*dw,u+eps*h,ref,phase)
        fm,_=m.system(omega-eps*dw,u-eps*h,ref,phase)
        numeric=np.r_[phase@h,(fp-fm)/(2*eps)]
        np.testing.assert_allclose(numeric,J@np.r_[dw,h],atol=4e-10,rtol=2e-7)
    def test_original_surrogate_coefficients_unchanged(self):
        d=self.d
        np.testing.assert_array_equal(d['u'],d['initial_u'])
        self.assertEqual(d['omega'].item(),d['initial_omega'].item())
        self.assertEqual(d['newton_steps'].item(),0)
        raw=loadmat(m.SOURCE)['candidate_real_fourier'].ravel();n=(len(d['u'])-1)//2
        pos=(raw[1::2]-1j*raw[2::2])/2
        source=np.r_[pos[:n].conj()[::-1],raw[0],pos[:n]]
        np.testing.assert_array_equal(d['u'].ravel(),source)
        self.assertTrue(np.all(pos[n:]==0))
    def test_reciprocal_partial_fraction_formula(self):
        x=np.linspace(.1,1.7,101);p=np.exp(1j*np.pi*(2*np.arange(10)+1)/10);c=1/(10*p**8)
        np.testing.assert_allclose(np.sum(c/(x[:,None]-p),axis=1),x/(1+x**10),atol=3e-15)
        np.testing.assert_allclose(np.sum(-c/(x[:,None]-p)**2,axis=1),(1-9*x**10)/(1+x**10)**2,atol=1e-14)
    def test_certificate_inequalities_and_physical_period(self):
        c=json.loads((HERE/'outputs_fixed_parameter/mg_fixed_parameter_certificate.json').read_text())
        self.assertTrue(c['proof_success'] and c['fixed_parameters'] and c['complete_infinite_tail'])
        self.assertEqual([c[k] for k in ('feedback','damping','delay','exponent')],[.2,.1,17,10])
        self.assertEqual(c['true_system_newton_steps'],0)
        self.assertLess(c['radii_polynomial_upper'],0);self.assertLess(c['contraction_upper'],1)
        self.assertLess(c['pole_neighborhood_factor'],1);self.assertGreater(c['nonconstant_coefficient_lower'],0)
        r=float(c['radius_decimal']);w=self.d['omega'].item()
        self.assertLessEqual(c['period_lower'],2*np.pi/(.1*(w+r)))
        self.assertGreaterEqual(c['period_upper'],2*np.pi/(.1*(w-r)))
if __name__=='__main__':unittest.main()
