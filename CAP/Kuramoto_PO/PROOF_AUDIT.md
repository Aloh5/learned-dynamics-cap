# Neumann KS proof audit

The a posteriori Fourier formulation follows Figueras and de la Llave,
[SIADS 16 (2017), 834–852](https://arxiv.org/abs/1605.01085).
The implementation uses an unweighted real trigonometric coefficient l1 norm,
not a coefficient supremum norm. The unknown correction norm is
`|delta f| + sum |delta v_nk|`, where `v=u/q`, `q=pi/22`, `s=q² t`,
and `f=omega/q²`. Temporal coefficients are constant/cosine/sine amplitudes.
The candidate and correction lie in the invariant subspace `n+k` even.
Only the time phase is fixed; the phase row uses the finite candidate tangent.

## Fixed-point operator

Let `L0=f*d_theta + nu*d_y^4 + d_y²`, `nu=q²`, `c=1/nu`, and
`Sc=L0+c`. The linearized augmented equation has finite block
`[0 phase; Sc^-1 v_theta, I-c Sc^-1+Sc^-1 d_y(v .)]`.
Use its stored approximate inverse B on the core (M=160, K=40), and L0^-1
on the state tail. B is interpreted as exact stored dyadic data: an inaccurate
numerical inverse cannot manufacture success because its defect is bounded.

The candidate has Mt=160 and Kx=60. Its quadratic residual is evaluated with
outward-rounded trigonometric transforms on 641 temporal nodes and 120 spatial
output modes. These retain the entire quadratic, including coefficients
outside the core. Y bounds the preconditioned residual. Alpha bounds the
linear fixed-point defect. Z2 bounds the quadratic remainder
`delta f * d_theta(delta v) + d_y(delta v²)/2` after preconditioning.
The real trigonometric l1 norm is a Banach algebra (the absolute coefficients
in each product-to-sum identity total at most one).

For radius r, `Y-(1-alpha)r+Z2*r²<0` and `alpha+2*Z2*r<1` give a contraction
mapping on the closed ball. The selected radius also lies strictly between
the quadratic roots. A nonzero temporal coefficient exceeding r excludes an
equilibrium. Positivity of the frequency enclosure gives the period bounds.

## Exhaustion of the infinite tail

All finite state columns AND the finite frequency column contribute to the
finite-to-tail bound. Frequency-to-tail is `L0^-1 v_theta` outside the core.
All allowed tail input columns through n=320, k=100 are checked: these are the
only ones that can couple into the finite block. Triangle estimates below
0.88 are retained; all other columns are multiplied by the full B.

Inputs 321<=n<360 are explicitly scanned. For n>=360 every output harmonic
is at least n-Mt>=200. The maximum of `(a+z)/(a²+z²)` over z>=z0 occurs at
`max(z0,(sqrt(2)-1)*a)`, with `a=|nu*k^4-k²|`. For spatial inputs k>=101,
all products remain outside the core and `k/(nu*k^4-k²)` decreases at the
relevant output modes k>=41. Thus the spatial scan's first endpoint already
bounds all larger inputs. The same monotonicity covers its infinite remainder.
The finite/tail derivative bounds used for Z2 cover all temporal and spatial
inputs by the same symbol maximization.

## Floating-point enclosure

The residual and final scalar inequalities use INTLAB intervals. The large
finite matrix and interface scan use IEEE binary64 arithmetic and explicit
forward-error bounds; they are not claimed to be interval matrix scans.
`gamma_m=m*u/(1-m*u)`, `u=2^-53`, is evaluated in INTLAB.

Unlike the former driver, matrix construction error is bounded using absolute
majorants BEFORE cancellation. Set U=||v||1, V=||v_theta||1 and
`a=(1+sqrt(2))/2`. Since `nu*k^4-k²+1/nu >= 3/(4*nu)`, set
`S=a*4*nu/3`. Let `D=max_k a*k/|nu*k^4-k²|`. The latter maximum is enclosed
for integers 1..2100 and decreases thereafter. The construction majorants are:

- finite matrix: `1+S/nu+S*K*U+2*V+S*V`;
- tail-to-core nonlinear column: `S*K*U`;
- raw nonlinear column, anywhere in the tail: `D*U`;
- frequency-to-tail: `a*U/f`;
- quadratic derivative operators: `1+||B||*S*(M+K/2)+a/f+D/2`.

A Fourier multiplication entry is a sum of at most six shifted coefficients;
real/imaginary conversion and shifts use exact powers of two. Symbols use
addition, subtraction, multiplication and division. The possible cancellation
condition `(nu*k^4+k²)/|nu*k^4-k²|` is explicitly checked to be <=4096; it
also bounds the raw symbol's componentwise parameter sensitivity. Shifted
symbols are better conditioned. Frequency is checked positive. Allowing 128
scalar operations/parameter-sensitivity factors per coefficient, plus fewer
than 2^18 summands in any direct-column norm, fits within the 2^20 gamma
budget. An additional factor eight covers composition of these bounds,
normalization by q, the real two-by-two symbols, and scalar maximizations.
All of these budgets apply to the fixed audited truncations; the MATLAB
driver rejects different truncations or interval lengths.

For dense matrix products the separate gamma_dimension bound multiplies
`||B||1 ||A_fl||1`; subtraction and the final column summations are bounded
separately. Construction error is propagated by `||B||1`. The two products
in the quadratic bound receive two matrix gamma terms. Tail-to-core matrix
products additionally use the largest triangle column sum from the scan.
All input constants in these guards are outward-rounded interval majorants.
No rounding guard assumes that cancellation makes intermediate terms small.

## Provenance and interpretation

The proof-data file stores the exact physical coefficients, frequency and
interval length used by the scan. MATLAB checks them against the candidate.
The exported polynomial has exact conjugate symmetry and exact `n+k` parity.
Canonicalizing a real Fourier representation is part of surrogate export,
not a true-system solve. The new generator never loads a true periodic orbit.
The only true-system evaluations are independent derivative training labels
and frozen-candidate diagnostics/proof. Surrogate optimizers may take many
steps; the true-system optimizer count remains zero throughout.

The fitted models use independent random spatial snapshots and exact spectral
derivative labels. They are complete quadratic models in sine coordinates,
which are invertibly related to interior spatial nodal values; this is not a
claim about learning these coefficients from noisy scalar observations.
Reflection equivariance is imposed by averaging the fitted polynomial with
`R f(R x)`. The selected periodic orbit is not used as training data.
