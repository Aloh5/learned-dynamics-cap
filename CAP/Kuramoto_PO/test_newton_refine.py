"""Numerical cross-checks of Fourier refinement, separate from the CAP."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.io import loadmat
from newton_refine import Refiner, complex_time, derivative, product_derivative, inverse_symbol
from diagnose_smooth_cap import real_time_coefficients

HERE=Path(__file__).resolve().parent

class NewtonFourierChecks(unittest.TestCase):
    def test_product_against_direct_coefficient_convolution(self):
        rng=np.random.default_rng(52)
        a=rng.normal(size=(9,4));b=rng.normal(size=(7,3))
        ac=complex_time(a);bc=complex_time(b)
        c=np.zeros((15,7),complex)
        for j in range(1,5):
            for k in range(1,4):
                value=np.convolve(ac[:,j-1],bc[:,k-1])
                c[:,j+k-1]+=.5*(j+k)*value
                if j!=k:c[:,abs(j-k)-1]-=.5*abs(j-k)*value
        expected=real_time_coefficients(c,7)
        np.testing.assert_allclose(product_derivative(a,b),expected,atol=2e-14,rtol=2e-14)

    def test_highest_temporal_product_is_not_aliased(self):
        # d_y[(cos(7 theta) sin y)^2] has temporal modes 0 and 14.
        a=np.zeros((15,1));a[7]=1
        result=product_derivative(a,a)
        self.assertAlmostEqual(result[0,1],.5,places=14)
        self.assertAlmostEqual(result[14,1],.5,places=14)
        result[0,1]=0;result[14,1]=0
        self.assertLess(np.max(abs(result)),1e-14)

    def test_linear_inverse_and_temporal_derivative(self):
        rng=np.random.default_rng(61);u=rng.normal(size=(11,8))
        f=4.2;nu=.0204;k=np.arange(1,9)
        L=f*derivative(u)+(nu*k**4-k*k)*u
        np.testing.assert_allclose(inverse_symbol(L,f,nu),u,atol=3e-15,rtol=3e-15)

    def test_quadratic_derivative_matches_centered_difference(self):
        rng=np.random.default_rng(11);a=rng.normal(size=(9,3));h=rng.normal(size=(9,3))
        eps=1e-5
        finite=(product_derivative(a+eps*h,a+eps*h)-product_derivative(a-eps*h,a-eps*h))/(4*eps)
        np.testing.assert_allclose(finite,product_derivative(a,h),atol=5e-10,rtol=5e-9)

    def test_executed_candidate_records_original_equation_updates(self):
        out=HERE/'outputs_newton_refined'
        if not (out/'candidate_diagnostics.json').exists():self.skipTest('No executed refined candidate.')
        d=json.loads((out/'candidate_diagnostics.json').read_text())
        self.assertEqual(d['true_system_newton_steps'],3)
        self.assertGreater(d['history'][0]['preconditioned_core_residual_l1']/d['history'][-1]['preconditioned_core_residual_l1'],100)
        u=loadmat(out/'ks_newton_refined_candidate.mat')['scaled_real_coefficients']
        m=(len(u)-1)//2;harm=np.r_[0,np.arange(1,m+1),np.arange(1,m+1)]
        self.assertTrue(np.all(u[(harm[:,None]+np.arange(1,u.shape[1]+1))%2!=0]==0))

    def test_original_inverse_matches_newton_core_jacobian_and_ordering(self):
        if not (HERE/'outputs_surrogate/ks_neumann_fourier_proof_data.mat').exists():
            self.skipTest('Regenerate the intentionally excluded original proof matrix; see README.')
        r=Refiner(160,60)
        rng=np.random.default_rng(21)
        h=np.zeros(r.dimension)
        h[0]=.1
        h[1+r.corepositions]=rng.normal(size=len(r.corepositions))
        h/=np.linalg.norm(h,1)
        action=r.jacobian_action(r.pack(r.f0,r.u0),h)
        select=np.r_[0,1+r.corepositions]
        np.testing.assert_allclose(action[select],h[select],atol=2e-11,rtol=2e-8)

    def test_executed_certificate_and_proof_dependency_hashes(self):
        import hashlib
        out=HERE/'outputs_newton_refined'
        if not (out/'newton_refinement_manifest.json').exists():self.skipTest('Run the interval workflow first.')
        c=json.loads((out/'ks_newton_refined_certificate.json').read_text())
        manifest=json.loads((out/'newton_refinement_manifest.json').read_text())
        self.assertTrue(c['pde_proof_success'] and c['tail_bound_complete'] and c['same_orbit_as_original'])
        self.assertTrue(c['phase_slice_identical'])
        self.assertEqual(c['newton_corrections'],3)
        self.assertLess(c['radius_polynomial'],0)
        self.assertLess(c['contraction_bound'],1)
        self.assertLessEqual(c['validation_radius'],7.000001e-9)
        self.assertLess(c['containment_upper'],c['original_radius'])
        packaging_path = out/'release_packaging.json'
        packaging = json.loads(packaging_path.read_text()) if packaging_path.exists() else {}
        for relative, expected in manifest['files_sha256'].items():
            with self.subTest(path=relative):
                path = HERE.parents[1]/relative
                if not path.exists() and path.name.endswith('fourier_proof_data.mat'):
                    self.skipTest('Regenerate the intentionally excluded proof matrix; see README.')
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected and relative in packaging:
                    entry = packaging[relative]
                    self.assertEqual(entry['executed_sha256'], expected)
                    expected = entry['release_sha256']
                self.assertEqual(actual, expected)


if __name__=='__main__':unittest.main()
