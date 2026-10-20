# Dense polynomial surrogate reconstruction

All baseline fits approximate derivative targets by an explicit vector field
`f(X) = C Θ(X)` in a complete monomial library. The shared implementation in
`dpsr/polynomial.py` fixes feature ordering, coefficient orientation, evaluation,
Jacobian calculation, coefficient export, and dense regression. This shares
the library-regression formulation of SINDy; iterative support selection is not
part of the baseline method. Pruning and feature augmentation are optional
ablations and must be reported separately.

| Setting | Coordinates | Training states and derivative targets | Estimator |
| --- | --- | --- | --- |
| MG forecasting | Delayed scalar observations | Trajectory windows; second-order finite differences | RMS-normalized, degree-weighted ridge; unpenalized intercept |
| Periodic KS forecasting | Periodic velocity nodes | Trajectory snapshots; exact dealiased spectral RHS | Unscaled dense quadratic least squares; zero regularization |
| Neumann KS construction | Sine coefficients, invertibly related to interior nodes | Independent paired random states; exact Galerkin RHS | Parity-separated dense quadratic least squares; reflection symmetrization |

## Numerical conventions

Ridge is solved as an augmented least-squares problem with penalty rows rather
than through normal equations. Polynomial features are ordered by total degree,
then combinations with replacement of coordinate indices. Coefficients have
shape `(output_dimension, feature_count)`. A polynomial model exports its
coefficients and exponent array to NPZ and supplies an analytic Jacobian.

The periodic KS fit deliberately does not inherit MG's column scaling.
Fixed-mean trajectories can make a polynomial library rank deficient; a change
of scaling changes the selected minimum-norm coefficient vector. The SVD solve
records rank, singular values, conditioning, and derivative residual. These
must be interpreted alongside rollout performance. A small derivative residual
alone does not establish accurate off-trajectory dynamics.

The Neumann fit keeps its original scaled coordinates and rank-revealing QR
solver (`gelsy`). Pairing states separates odd and even features; symmetry
averaging follows regression. It uses the shared dense solver while retaining
its specialized quadratic tensor representation for proof construction.

## Assessment and claims

Split training, validation and test data before tuning. If an existing sweep's
held-out windows are used to choose a model, those windows are validation data;
obtain an untouched test window before making a new test-performance claim.
The refactor preserves historical baseline configurations and does not claim
to have rerun or redesigned every experimental sweep.

MG uses the first observation coordinate for forecast errors; KS uses spatial
mean squared error. Their cumulative-error thresholds remain 0.05 and 0.5,
respectively. These metrics support comparisons within each setting.

Forecasting, invariant-object discovery, and rigorous validation are separate
stages. Exact derivative labels from a known semidiscretization constitute a
different information regime from derivative estimates of scalar observations.
Periodic and Neumann KS have different boundary conditions. Numerical
regression tests do not prove PDE approximation, chaos, or orbital stability.

Frozen proof artifacts remain tied to their original coefficients and
candidates. Replacing numerical solvers or changing training data does not
transfer an existing certificate to a newly generated candidate.
