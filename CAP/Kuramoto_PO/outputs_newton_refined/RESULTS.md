# Newton-refined Neumann KS result

MATLAB R2024b / INTLAB 13 validated an ordinary periodic solution of the
original Neumann KS equation on a length-22 interval.

| Quantity | Result |
|---|---:|
| Original certified radius | 1.000000000000e-05 |
| Refined certified radius | 7.000000000000e-09 |
| Radius improvement factor | 1428.57 |
| True-equation Newton updates | 3 |
| Candidate temporal / spatial orders | 220 / 100 |
| Original fixed inverse core orders | 160 / 40 |
| Initial numerical preconditioned core residual | 4.883484296040e-10 |
| Final numerical preconditioned core residual | 1.873448914751e-13 |
| Certified preconditioned residual Y | 6.964318201318e-10 |
| Certified derivative defect alpha | 8.908095064589e-01 |
| Global quadratic bound Z2 | 1.650850833256e+02 |
| Radii polynomial upper bound | -6.789354548707e-11 |
| Contraction upper bound | 8.908118176500e-01 |
| Center displacement upper bound | 4.915051273093e-10 |
| Refined ball distance from original center, upper bound | 7.491505127309e-09 |
| Physical period lower endpoint | 72.70550486116962 |
| Physical period upper endpoint | 72.70550510134991 |

The norm is |delta f| plus the l1 sum of real constant/cosine/sine velocity
amplitudes, with v=u/q, q=pi/22, f=omega/q^2. This is the same unweighted
real trigonometric norm as the original Neumann certificate. It differs
from the weighted Mackey–Glass norm; radii are not cross-system rankings.

The full-space derivative bound is transferred from the original fixed
operator using alpha_new <= alpha_original + 2 Z2 d. The original finite
inverse, smoothing, tail inverse, and phase row remain fixed. A fresh scan
at the new center is not asserted. The new residual is enclosed by direct
interval temporal convolutions and exact sine product identities, retaining
all modes through time 440 and space 200.
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
