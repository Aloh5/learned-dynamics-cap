"""Independent numerical checks; only the INTLAB driver establishes existence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mg_candidate', HERE/'fixed_parameter_candidate.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class NewtonRefinementChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name)
        subprocess.run([sys.executable, str(HERE/'fixed_parameter_candidate.py'),
                        '--newton-steps', '5', '--output-dir', str(cls.out)],
                       check=True, capture_output=True)
        cls.d = loadmat(cls.out/'mg_fixed_parameter_candidate.mat')
        cls.record = json.loads((cls.out/'candidate_diagnostics.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_newton_reduces_residual_and_preserves_phase(self):
        h = self.record['history']
        self.assertLess(h[-1]['residual_l1'], 1e-14)
        self.assertGreater(h[0]['residual_l1']/h[-1]['residual_l1'], 1e10)
        d = self.d
        self.assertLess(abs((d['phase']@d['u']).item()-d['phase_target'].item()), 1e-15)
        np.testing.assert_array_equal(d['u'], d['u'][::-1].conj())

    def test_independent_oversampled_original_dde_residual(self):
        # Direct derivative and shifted Fourier waveform; no reciprocal/partial
        # fraction approximation and twice the candidate builder's FFT grid.
        d = self.d
        u = d['u'].ravel()
        n = (len(u)-1)//2
        k = np.arange(-n, n+1)
        omega = d['omega'].item()
        values = m.evaluate(u, size=16384)
        delayed = m.evaluate(u*np.exp(-1j*k*omega*1.7), size=16384)
        derivative = m.evaluate(1j*k*omega*u, size=16384)
        residual = derivative + values - 2*delayed/(1+delayed**10)
        self.assertLess(np.max(abs(residual)), 2e-13)

    def test_correction_remains_inside_original_certified_neighborhood_numerically(self):
        distance = self.record['frequency_change'] + self.record['state_change_weighted_l1']
        self.assertLess(distance, 0.0006)
        self.assertGreater(distance, 1e-4)

    def test_original_certificate_cannot_be_overwritten_by_newton_cli(self):
        baseline = HERE/'outputs_fixed_parameter/mg_fixed_parameter_certificate.json'
        before = baseline.read_bytes()
        r = subprocess.run([sys.executable, str(HERE/'fixed_parameter_candidate.py'),
                            '--newton-steps', '1'], capture_output=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(before, baseline.read_bytes())

    def test_executed_refinement_certificate_and_artifact_hashes(self):
        import hashlib
        out = HERE/'outputs_newton_refined'
        if not (out/'newton_refinement_manifest.json').exists():
            self.skipTest('Run the INTLAB refinement workflow to produce proof artifacts.')
        cert = json.loads((out/'mg_fixed_parameter_certificate.json').read_text())
        summary = json.loads((out/'newton_refinement_summary.json').read_text())
        manifest = json.loads((out/'newton_refinement_manifest.json').read_text())
        self.assertTrue(cert['proof_success'] and cert['complete_infinite_tail'])
        self.assertTrue(summary['same_orbit_as_original'] and summary['phase_slice_identical'])
        self.assertEqual(cert['true_system_newton_steps'], 5)
        self.assertLessEqual(cert['radius'], 2e-12)
        self.assertLess(cert['radii_polynomial_upper'], 0)
        self.assertLess(cert['contraction_upper'], 1)
        self.assertLess(summary['refined_ball_distance_from_original_upper'], 0.0006)
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


if __name__ == '__main__':
    unittest.main()
