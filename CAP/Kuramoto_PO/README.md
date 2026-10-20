# Kuramoto--Sivashinsky Neumann periodic-orbit proof

This directory contains the numerical and computer-assisted proof workflow for
an ordinary periodic orbit of the true Kuramoto--Sivashinsky PDE with fixed
endpoint conditions. It is separate from the periodic-domain, relative-orbit
experiment in `notebooks/kuramoto_sivashinsky/ks_periodic_forecasting.ipynb`.
`ks_neumann_orbit_certification.ipynb` explains and exercises this formulation.

## Formulation and symmetry

On `0 <= x <= ell`, the mean-zero potential satisfies

```text
h_t = -(1/2) h_x^2 - h_xx - h_xxxx + mean_x((1/2) h_x^2),
h_x(0,t) = h_x(ell,t) = 0,
h_xxx(0,t) = h_xxx(ell,t) = 0.
```

With `u=h_x`, this is exactly

```text
u_t = -u u_x - u_xx - u_xxxx,
u(0,t) = u(ell,t) = 0,
u_xx(0,t) = u_xx(ell,t) = 0.
```

Thus `h` uses a cosine basis and `u` a sine basis. After `y=pi*x/ell`,
`v=u/(pi/ell)`, and `s=(pi/ell)^2 t`, the equation is

```text
v_s + v v_y + v_yy + nu v_yyyy = 0,
nu = (pi/ell)^2.
```

A generic spatial translate does not preserve the odd extension or the fixed
endpoints. The continuous translation group is therefore absent, so the proof
solves for an ordinary periodic orbit and retains only a temporal phase
condition. There is no drift variable, spatial shift, or spatial phase
condition. A discrete equivariance remains:

```text
R u(x) = -u(ell-x),       R^2 = I.
```

The validated orbit has the spatio-temporal symmetry
`u(theta+pi,x)=R u(theta,x)`. In Fourier variables this is `n+k` even. The
proof is carried out in that closed invariant subspace; this halves the finite
matrix dimension without reintroducing a continuous neutral direction.

## Surrogate-only workflow (current result)

`generate_surrogate_candidate.py` implements the requested sequence:

1. Fit complete quadratic surrogate ODEs from independent random spatial
   snapshots with exact spectral derivative labels. Paired positive/negative
   snapshots separate odd and even features in full-rank least squares.
   Reflection equivariance is imposed on the fitted polynomial. No true
   periodic orbit is used for training or initialization.
2. Integrate the 16-mode **surrogate** and scan its trajectory for close returns.
3. Refine the orbit only in fitted surrogates, continuing through
   `(temporal, spatial)` sizes `(16,16), (24,20), (40,28), (64,36), (80,40),
   `(96,44), (128,52), (160,60)`. KS is quadratic, so spatial/temporal
   resolution increases while the polynomial degree stays two.
4. Canonicalize reality and half-period symmetry during surrogate export;
   this changes coefficients by at most `6.47e-14`. Freeze the polynomial.
5. Evaluate and validate this frozen candidate in the true infinite-dimensional
   equation. Apply **zero Newton corrections and zero optimizer calls in the
   true system**, including candidate construction.

The coordinates are sine coefficients, invertibly related to interior spatial
nodal values. This is a data-fitted model with exact derivative supervision;
it is not evidence for learning the same orbit from noisy scalar KS data.
The 60-mode model uses 7,324 training snapshots and 512 independent held-out
snapshots. Its held-out derivative relative L2 error is `5.98e-16` (rounded up).

Current frozen candidate:

```text
ell                         22
period                      72.70550498125212
true oversampled RMS        2.42093e-13  (rounded up)
independent closure L2      2.91707e-11  (rounded up)
true-system optimizer calls 0
true-system Newton steps    0
```

Outputs are in `outputs_surrogate/`, including the final 60-mode fitted model,
surrogate trajectory, recurrence table, final continuation checkpoint, metrics,
and proof certificate. Earlier model and continuation checkpoints are under
`archive/CAP/Kuramoto_PO/`. The notebook `ks_neumann_orbit_certification.ipynb` uses the
active outputs by default.

The **legacy true-Galerkin candidate** workflow is preserved under
`archive/CAP/Kuramoto_PO/`. Its zero count applies only after export; it does not establish
surrogate-only transfer. Do not use that legacy certificate as evidence for
the new zero-true-correction claim.

## Infinite-dimensional proof

The certificate uses the smoothed space-time Fourier correction equation of
Figueras and de la Llave. The finite block contains temporal harmonics
`0,...,160`, spatial sine modes `1,...,40`, the frequency variable, and the
temporal phase equation. Enforcing `n+k` even gives dimension 6421.

`scan_block_proof.py` constructs an approximate inverse of that finite block
using direct Fourier shifts. On the complementary tail it uses the exact
diagonal inverse of `f*d_theta + nu*d_y^4 + d_y^2`. It then checks:

- every finite-to-tail column;
- every tail-to-finite and tail-to-tail column in the complete coupling halo
  `n <= 320`, `k <= 100`;
- the remaining temporal gap `321 <= n < 360`;
- analytic continuous-symbol bounds for `n >= 360`; and
- analytic spatial bounds for `k >= 101`.

Columns whose triangle bound is at least 0.88 are recomputed with the full
finite inverse. The worst interface column is temporal mode 161, spatial mode
17, and is bounded by 0.890514245 before rounding guards. This direct audit
uses the correct Fourier convolution norm; it does not rely on the inconsistent
coefficient-supremum expression printed for `K3` in one line of the reference
paper.

`proof_KS_neumann.m` uses INTLAB outward rounding to recompute the fully
de-aliased residual on 641 temporal nodes and 120 spatial output modes. It
encloses all final inequalities, including conservative Higham `gamma_n`
forward-error guards for the regenerated double-precision finite operator and
interface scan.

The current surrogate-only certificate is

```text
Y              <= 4.119405e-7
alpha          <= 0.890809345
Z2             <= 165.085084
chosen radius       1e-5
radii polynomial <= -6.634575e-7
contraction     <= 0.894111046
period lies in     [72.70533342439506, 72.70567653891881]
```

The radius is in the scaled Fourier correction norm defined in
[PROOF_AUDIT.md](PROOF_AUDIT.md). A nonzero temporal coefficient excludes an
equilibrium. The certificate reports

```text
candidate_origin              = data_fitted_surrogate_only
true_system_optimizer_calls   = 0
newton_corrections             = 0
finite_galerkin_proof_success  = true
tail_bound_complete           = true
pde_proof_success             = true
```

The proof explicitly includes the frequency-to-tail column, checks the
contraction constant, and rejects proof data whose coefficients, frequency or
interval length do not match the candidate. Large matrix computations use
IEEE forward-error bounds based on absolute intermediate majorants; INTLAB
encloses the residual and final inequalities. See the audit for the norm,
operator, complete tail argument, rounding assumptions and operation budget.

## Reproduction

Python requires NumPy, SciPy, and pandas. MATLAB R2024b and INTLAB 13 were
used for the outward-rounded stage. From the repository root:

```bash
bash CAP/Kuramoto_PO/run_surrogate_pde_proof.sh
```

The driver resumes available surrogate fits and continuation checkpoints;
missing intermediate checkpoints are regenerated. For an independent clean
candidate run, use a new output directory:

```bash
python CAP/Kuramoto_PO/generate_surrogate_candidate.py --output-dir /tmp/ks_clean
```

`MATLAB_BINARY` and `INTLAB_DIR` can override the installed paths. The finite
inverse MAT file is about 307 MB and is generated, not source code. CSV/JSON
certificates, stage summaries and fit metrics are small inspectable artifacts.
A candidate diagnostic file intentionally says the generator has not evaluated
the proof; the proof outcome is in `ks_neumann_full_pde_certificate.json`.

Validation completed: 11 unit tests passed, the notebook default workflow ran,
and MATLAB/INTLAB produced the certificate above. The legacy driver and outputs
remain recoverable from the dated cleanup archive.

## Source files

- `ks_neumann.py`: basis, ETDRK4 flow, recurrence search, collocation system,
  temporal phase condition, and diagnostics.
- `surrogate_neumann.py`: fitted quadratic models and independent fit diagnostics.
- `generate_surrogate_candidate.py`: surrogate recurrence, refinement and frozen export.
- `run_surrogate_pde_proof.sh`: current complete reproduction driver.
- `diagnose_smooth_cap.py`: direct smoothed-operator and tail-column routines.
- `scan_block_proof.py`: exhaustive finite/tail proof-data construction.
- `proof_KS_neumann.m`: outward-rounded residual, radii polynomial, and
  certificate.
- `test_surrogate_neumann.py`: fit, Jacobian, symmetry and learned-RHS routing tests.
- `test_ks_neumann.py`: basis, boundary, nonlinearity, Fourier, symmetry, and
  direct-column tests.

Legacy true-Galerkin, CAPD, and intermediate continuation files are preserved
under `archive/CAP/Kuramoto_PO/`.

Primary mathematical reference: J.-L. Figueras and R. de la Llave,
“Numerical Computations and Computer Assisted Proofs of Periodic Orbits of the
Kuramoto--Sivashinsky Equation,” SIAM J. Applied Dynamical Systems 16 (2017),
834--852, https://arxiv.org/abs/1605.01085.
