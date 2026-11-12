"""Report executed certificates; this script does not replace interval proof."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs_newton_refined'

def main():
    cert=json.loads((OUT/'mg_fixed_parameter_certificate.json').read_text())
    summary=json.loads((OUT/'newton_refinement_summary.json').read_text())
    diag=json.loads((OUT/'candidate_diagnostics.json').read_text())
    old=json.loads((HERE/'outputs_fixed_parameter/mg_fixed_parameter_certificate.json').read_text())
    assert cert['proof_success'] and summary['proof_success'] and summary['same_orbit_as_original']
    assert cert['radius_decimal']==summary['refined_radius_decimal']
    assert cert['true_system_newton_steps']==diag['true_system_newton_steps']==summary['true_system_newton_steps']
    assert cert['radii_polynomial_upper']<0 and cert['contraction_upper']<1
    candidate=loadmat(OUT/'mg_fixed_parameter_candidate.mat')
    assert candidate['newton_steps'].item()==diag['true_system_newton_steps']
    d=diag['history'][-1]
    files=[HERE/'fixed_parameter_candidate.py',HERE/'proof_MG_fixed_parameter.m',
           HERE/'proof_MG_newton_refined.m',HERE/'run_newton_refined_proof.sh',
           HERE/'summarize_newton_refinement.py',HERE/'test_newton_refinement.py',
           HERE/'outputs_fixed_parameter/mg_fixed_parameter_candidate.mat',
           HERE/'outputs_fixed_parameter/mg_fixed_parameter_certificate.json',
           OUT/'mg_fixed_parameter_candidate.mat',OUT/'mg_fixed_parameter_certificate.json',
           OUT/'mg_fixed_parameter_bounds.mat',OUT/'newton_refinement_containment.mat',
           OUT/'newton_refinement_summary.json',OUT/'candidate_diagnostics.json',
           HERE.parents[1]/'notebooks/mackey_glass/outputs_zero_correction/MG_zero_correction_candidate.mat']
    hashes={str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest={'scope':'Fixed-parameter Mackey-Glass only; no claim of KS refinement',
              'numerical_core_residual_l1':d['residual_l1'],
              'original_certified_radius':old['radius'],
              'refined_certified_radius':cert['radius'],
              'certified_radius_improvement_factor':old['radius']/cert['radius'],
              'files_sha256':hashes}
    (OUT/'newton_refinement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=f'''# Newton-refined Mackey–Glass periodic-orbit certificate

Executed with MATLAB R2024b and INTLAB 13. This is a rigorous existence result
for the original delay equation, not merely a numerical residual calculation.

| Quantity | Result |
|---|---:|
| True-equation Newton updates | {cert['true_system_newton_steps']} |
| Initial numerical core residual (unweighted l1) | {diag['history'][0]['residual_l1']:.12e} |
| Final numerical core residual (unweighted l1) | {d['residual_l1']:.12e} |
| Certified preconditioned residual bound Y | {cert['Y_upper']:.12e} |
| Certified linear defect Z0 | {cert['Z0_upper']:.12e} |
| Certified derivative-variation bound C | {cert['C_upper']:.12e} |
| Original certified radius | {old['radius']:.12e} |
| Refined certified radius | {cert['radius']:.12e} |
| Radius improvement factor | {manifest['certified_radius_improvement_factor']:.6g} |
| Radii polynomial upper bound | {cert['radii_polynomial_upper']:.12e} |
| Contraction upper bound | {cert['contraction_upper']:.12e} |
| Center displacement upper bound | {summary['center_distance_upper']:.12e} |
| Refined ball distance from original center, upper bound | {summary['refined_ball_distance_from_original_upper']:.12e} |
| Certified physical period, lower endpoint | {cert['period_lower']:.16g} |
| Certified physical period, upper endpoint | {cert['period_upper']:.16g} |

The radius uses |delta omega| + sum_k |delta u_k| (1.01)^|k|, where
s=t/10 and physical angular frequency is omega/10. It bounds frequency and
waveform together, not the physical period directly. The interval period is
reported separately. The same phase slice is used in both certificates.
Interval containment in the original radius-0.0006 ball identifies the new
solution with the originally certified solution by that theorem's uniqueness.

All physical parameters remain fixed: feedback 0.2, damping 0.1, delay 17,
and exponent 10. The candidate/inverse order is 80, reciprocal order 160,
and derivative polynomial order 160. Infinite tails, reciprocal errors,
nonstationarity, phase, and finite inverse defect are covered by the proof.

The numerical residual and the certified distance are different quantities.
The final interval residual enclosure, including reciprocal and arithmetic
errors, sets the practical accuracy reached by this implementation. This run
does not claim a certified 1e-13 radius, orbital stability, a minimal period,
chaos, or global equivalence. It certifies one Mackey–Glass periodic solution;
no Newton-refined KS certificate is claimed.

Reproduce from the repository root:

```sh
PYTHON_BINARY=.venv/bin/python bash CAP/Mackey-Glass_PO/run_newton_refined_proof.sh
```

MATLAB_BINARY, INTLAB_PATH, PYTHON_BINARY, and MG_CAP_RADIUS can be overridden.
The original zero-correction artifacts are preserved in outputs_fixed_parameter/.
See newton_refinement_manifest.json for SHA256 hashes of proof inputs and code.
'''
    (OUT/'RESULTS.md').write_text(report)
    lo=np.nextafter(cert['period_lower'],-np.inf)
    hi=np.nextafter(cert['period_upper'],np.inf)
    theorem=rf'''% Insert after the uncorrected Mackey--Glass theorem in Section 6.
% Generated only after the executed interval certificate and containment passed.
\begin{{theorem}}[Newton-refined Mackey--Glass periodic orbit]
\label{{thm:mgorbit-refined}}
Let $(\widetilde\omega,\widetilde u)$ be the candidate obtained by five
Newton updates in the phase-augmented, fixed-parameter Mackey--Glass
Fourier equation, starting from the candidate of Theorem~\ref{{thm:mgorbit}}.
The same nonstationary periodic solution satisfies
\[
 |\omega_*-\widetilde\omega|+
 \|u_*-\widetilde u\|_{{1.01}}\leq {cert['radius_decimal']}.
\]
It is unique in this ball and the stated phase slice. Its physical period
belongs to $[{lo:.16g},\;{hi:.16g}]$.
\end{{theorem}}
\begin{{proof}}
Outward-rounded INTLAB calculations give
$Y\leq {cert['Y_upper']*1.000001:.9e}$,
$Z_0\leq {cert['Z0_upper']*1.000001:.9e}$, and
$C\leq {cert['C_upper']*1.000001:.9e}$.
At $r={cert['radius_decimal']}$, the validated radii polynomial is negative
and the contraction bound is less than one. The proof includes all Fourier
tails, pole exclusion, nonstationarity, and injectivity of the approximate
inverse, so Proposition~\ref{{prop:radii}} applies.
The phase condition is identical to that of the uncorrected certificate.
Interval arithmetic bounds the distance from the original center to the
entire refined ball by ${summary['refined_ball_distance_from_original_upper']*1.000001:.9e}
<6.00\times10^{{-4}}$. Uniqueness in the original ball identifies the two
certified solutions. The period enclosure follows from outward-rounded
frequency conversion.
\end{{proof}}
'''
    # Use conventional TeX scientific notation, including the radius.
    import re
    theorem=re.sub(r'([0-9]+(?:\.[0-9]+)?)e([+-]?\d+)',
                   lambda m:m.group(1)+r'\times10^{'+str(int(m.group(2)))+'}',theorem)
    (OUT/'theorem_newton_refined.tex').write_text(theorem)
    print(report)

if __name__=='__main__':main()
