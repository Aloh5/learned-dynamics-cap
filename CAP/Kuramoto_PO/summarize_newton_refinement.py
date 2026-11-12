"""Summarize executed interval results and seal their artifact dependency chain."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs_newton_refined'
BASE=HERE/'outputs_surrogate'

def main():
    c=json.loads((OUT/'ks_newton_refined_certificate.json').read_text())
    d=json.loads((OUT/'candidate_diagnostics.json').read_text())
    old=json.loads((BASE/'ks_neumann_full_pde_certificate.json').read_text())
    candidate=loadmat(OUT/'ks_newton_refined_candidate.mat')
    assert c['pde_proof_success'] and c['tail_bound_complete'] and c['same_orbit_as_original']
    assert c['radius_polynomial']<0 and c['contraction_bound']<1
    assert c['newton_corrections']==d['true_system_newton_steps']==candidate['newton_steps'].item()
    assert c['containment_upper']<old['validation_radius']
    assert d['source_sha256']==hashlib.sha256((BASE/'ks_neumann_cap_candidate.mat').read_bytes()).hexdigest()
    assert c['temporal_modes']==candidate['temporal_modes'].item()
    assert c['spatial_modes']==candidate['spatial_modes'].item()
    paths=[HERE/'newton_refine.py',HERE/'proof_KS_newton_refined.m',
           HERE/'run_newton_refined_proof.sh',HERE/'summarize_newton_refinement.py',
           HERE/'NEWTON_REFINEMENT.md',HERE/'test_newton_refine.py',
           HERE/'proof_KS_neumann.m',HERE/'scan_block_proof.py',HERE/'diagnose_smooth_cap.py',HERE/'PROOF_AUDIT.md',
           BASE/'ks_neumann_cap_candidate.mat',BASE/'ks_neumann_full_pde_certificate.json',
           BASE/'ks_neumann_fourier_proof_data.mat',OUT/'ks_newton_refined_candidate.mat',
           OUT/'ks_newton_refined_certificate.json',OUT/'ks_newton_refined_bounds.mat',OUT/'candidate_diagnostics.json']
    manifest={'scope':'Neumann KS only; periodic-domain forecasting is a different problem',
              'certificate_radius':c['validation_radius'],
              'original_radius':old['validation_radius'],
              'radius_improvement_factor':old['validation_radius']/c['validation_radius'],
              'files_sha256':{str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'newton_refinement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=f'''# Newton-refined Neumann KS result

MATLAB R2024b / INTLAB 13 validated an ordinary periodic solution of the
original Neumann KS equation on a length-22 interval.

| Quantity | Result |
|---|---:|
| Original certified radius | {old['validation_radius']:.12e} |
| Refined certified radius | {c['validation_radius']:.12e} |
| Radius improvement factor | {manifest['radius_improvement_factor']:.6g} |
| True-equation Newton updates | {c['newton_corrections']} |
| Candidate temporal / spatial orders | {c['temporal_modes']} / {c['spatial_modes']} |
| Original fixed inverse core orders | 160 / 40 |
| Initial numerical preconditioned core residual | {d['history'][0]['preconditioned_core_residual_l1']:.12e} |
| Final numerical preconditioned core residual | {d['history'][-1]['preconditioned_core_residual_l1']:.12e} |
| Certified preconditioned residual Y | {c['Y']:.12e} |
| Certified derivative defect alpha | {c['alpha']:.12e} |
| Global quadratic bound Z2 | {c['Z2']:.12e} |
| Radii polynomial upper bound | {c['radius_polynomial']:.12e} |
| Contraction upper bound | {c['contraction_bound']:.12e} |
| Center displacement upper bound | {c['center_distance_upper']:.12e} |
| Refined ball distance from original center, upper bound | {c['containment_upper']:.12e} |
| Physical period lower endpoint | {c['true_period_lower']:.16g} |
| Physical period upper endpoint | {c['true_period_upper']:.16g} |

The norm is |delta f| plus the l1 sum of real constant/cosine/sine velocity
amplitudes, with v=u/q, q=pi/22, f=omega/q^2. This is the same unweighted
real trigonometric norm as the original Neumann certificate. It differs
from the weighted Mackey–Glass norm; radii are not cross-system rankings.

The full-space derivative bound is transferred from the original fixed
operator using alpha_new <= alpha_original + 2 Z2 d. The original finite
inverse, smoothing, tail inverse, and phase row remain fixed. A fresh scan
at the new center is not asserted. The new residual is enclosed by direct
interval temporal convolutions and exact sine product identities, retaining
all modes through time {c['residual_temporal_output_modes']} and space {c['residual_spatial_output_modes']}.
This replaces the conservative trigonometric residual evaluation used by the
original certificate. Newton refinement, increased candidate orders, and
sharper interval residual evaluation all contribute to the result; their
individual effects have not been isolated by a controlled comparison.

Interval containment and the unchanged phase/symmetry slice identify the
same orbit as in the original certificate. All omitted correction modes,
arithmetic errors, nonstationarity, and positive frequency are covered.
Numerical residual and certified distance are different quantities. No
orbital stability, minimal period, chaos, or global equivalence is proved.
The original zero-correction certificate is preserved. This result does not
certify the periodic-domain KS forecasting experiment.

Reproduce:

```sh
PYTHON_BINARY=.venv/bin/python bash CAP/Kuramoto_PO/run_newton_refined_proof.sh
```

See ../NEWTON_REFINEMENT.md for the proof argument and
newton_refinement_manifest.json for the hashed dependency chain.
'''
    (OUT/'RESULTS.md').write_text(report)
    print(report)

if __name__=='__main__':main()
