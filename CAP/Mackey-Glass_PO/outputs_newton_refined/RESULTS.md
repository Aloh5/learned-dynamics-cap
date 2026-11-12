# Newton-refined Mackey–Glass periodic-orbit certificate

Executed with MATLAB R2024b and INTLAB 13. This is a rigorous existence result
for the original delay equation, not merely a numerical residual calculation.

| Quantity | Result |
|---|---:|
| True-equation Newton updates | 5 |
| Initial numerical core residual (unweighted l1) | 4.713396703762e-04 |
| Final numerical core residual (unweighted l1) | 9.145974360733e-17 |
| Certified preconditioned residual bound Y | 1.033277460233e-12 |
| Certified linear defect Z0 | 7.000642564048e-02 |
| Certified derivative-variation bound C | 4.894484523452e+02 |
| Original certified radius | 6.000000000000e-04 |
| Refined certified radius | 1.200000000000e-12 |
| Radius improvement factor | 5e+08 |
| Radii polynomial upper bound | -8.271482864651e-14 |
| Contraction upper bound | 7.000642622781e-02 |
| Center displacement upper bound | 4.194907094594e-04 |
| Refined ball distance from original center, upper bound | 4.194907106594e-04 |
| Certified physical period, lower endpoint | 46.72953965885939 |
| Certified physical period, upper endpoint | 46.72953965894284 |

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
