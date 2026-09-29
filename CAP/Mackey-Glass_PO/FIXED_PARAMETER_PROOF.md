# Fixed-parameter Mackey–Glass periodic-orbit proof

The executed certificate is `outputs_fixed_parameter/mg_fixed_parameter_certificate.json`.
The physical equation is **exactly**

\[
x'(t)=\frac{0.2x(t-17)}{1+x(t-17)^{10}}-0.1x(t).
\]

Feedback, damping, delay, and exponent are fixed. The unknown scalar is frequency. The original surrogate waveform is unchanged, including its phase; the exported 200-mode array has exactly zero coefficients beyond mode 40. The proof uses an 80-mode finite inverse. No true-equation Newton correction is applied in the certified run.

## Reproduction

From the repository root:

```sh
PYTHON_BINARY=/opt/miniconda3/bin/python bash CAP/Mackey-Glass_PO/run_fixed_parameter_proof.sh
```

Python requires NumPy and SciPy. MATLAB and INTLAB 13 are required for the proof. `MATLAB_BINARY`, `INTLAB_DIR`, and `PYTHON_BINARY` override machine-specific defaults. The focused notebook is `notebooks/mackey_glass/Mackey_Glass_Fixed_Parameter_CAP.ipynb`. Its default mode checks and displays the executed certificate; set `MG_RUN_FIXED_PARAMETER_PROOF=1` to regenerate it.

The candidate builder performs only floating-point preparation. Its Fourier approximations and numerical inverse are interpreted as exact dyadic input data by the interval proof; their quality is verified, never assumed. An optional Newton setting is for numerical investigation. The reproduction script explicitly selects zero, and the notebook/tests verify equality with the source coefficients.

## Space, phase, and operator

Let s=t/10, tau=17/10, G(z)=z/(1+z^10), and theta=omega s. Write
u(theta)=sum u_k exp(ik theta), with conjugate-symmetric coefficients in l1_nu, nu=101/100. Use the real Banach space

\[
X=\mathbb R\times\ell^1_{\nu,\mathrm{real}},\qquad
\|(\delta\omega,h)\|_X=|\delta\omega|+\sum_k|h_k|\nu^{|k|}.
\]

The positive constant reference frequency omega_ref is the exact binary64 candidate frequency. Its use only rescales equations; it is not a free physical parameter. Define S_omega a_k=exp(-ik omega tau)a_k, P as removal of mode zero, and J_0=1, J_k=1/(ik omega_ref) for k != 0. The state residual is

\[
F_u(\omega,u)=P u+\frac{\omega-\omega_{\rm ref}}{\omega_{\rm ref}}Pu+J(u-2S_\omega G(u)).
\]

Thus F_0=u_0-2G(u)_0, and every nonzero equation is equivalent to
ik omega u_k+u_k-2 exp(-ik omega tau)G(u)_k=0.
The phase residual is phi u - b. The finite row phi is the dyadic approximation to the normalized conjugate temporal tangent of the input, with support |k|<=80, exact conjugate symmetry, and the target b stored as exact dyadic data. This finite linear phase is used identically in the numerical inverse and the interval residual. Its rounding residual is included in Y.

The derivatives are

\[
D_uF_u=\frac{\omega}{\omega_{\rm ref}}P+J(I-2S_\omega M_{G'(u)}),\qquad
D_\omega F_u=\frac1{\omega_{\rm ref}}P(u+2\tau S_\omega G(u)).
\]

The first displayed state formula contains P u, **not u**; zero mode is supplied only by J. Code sets the zero row explicitly. Frequency is real. Real shifts are isometries on l1_nu. The integrated shift is continuously differentiable in frequency even though the unsmoothed shift is not operator-norm continuous. The bounds below split changes into a state perturbation and a shift of the smooth, finite-center nonlinear waveform; they do not assume a bounded temporal derivative on the whole Banach ball.

## Rational nonlinearity and pole exclusion

Let p_j=exp(i pi(2j-1)/10), c_j=1/(10p_j^8), j=1,...,10. The exact partial fraction identity is

\[
G(z)=\sum_j c_j(z-p_j)^{-1}.
\]

For each pole, a degree-160 approximate reciprocal b_j is supplied as dyadic Fourier data. Direct interval convolution (no FFT in the proof) encloses

\[
e_j=\|1-(\bar u-p_j)b_j\|_\nu<1,\quad
B_j=\frac{\|b_j\|_\nu}{1-e_j},\quad d_j=B_j e_j.
\]

The Banach-algebra Neumann series proves that R_j=(bar u-p_j)^(-1) exists with norm <= B_j and ||R_j-b_j|| <= d_j. Poles and residues are enclosed with INTLAB pi and complex exponential. The center approximations are g=sum c_j b_j and g'= -sum c_j b_j*b_j. Their total errors are bounded by sum |c_j|d_j and sum |c_j|d_j(B_j+||b_j||), respectively. The latter polynomial is truncated to degree P=160 and **the entire discarded weighted norm is added to its error**. All convolution and truncation roundoff is already interval-enclosed.

For every ||h||<=r, B_j r<1 proves invertibility of bar u+h-p_j on the entire ball. Define

\[
L_r=\sum_j |c_j|\left(\frac{B_j}{1-B_jr}\right)^2,\qquad
W_r=2\sum_j |c_j|\left(\frac{B_j}{1-B_jr}\right)^3.
\]

These bound ||G'(bar u+h)|| and ||G''(bar u+h)||, so ||G'(bar u+h)-G'(bar u)||<=W_r||h||. There is no smaller unverified neighborhood: the driver checks each B_j r<1 at the very radius used in the contraction inequalities.

## Approximate inverse and all infinite columns

A is the numerical inverse on the augmented core |k|<=N=80 and multiplication by the positive dyadic constant a_tail=omega_ref/bar omega on its state complement. In the zero-correction run a_tail=1 exactly. The input matrices preserve conjugate symmetry, checked by exact array equality.

The interval calculation includes input modes |j|<=M=N+P=240 and output modes |k|<=K=N+2P=400. This contains the full multiplication support for the retained derivative polynomial. The phase row has no support beyond N. The frequency column includes its entire retained support, and its nonlinear approximation error is separately bounded.

For |j|>M, multiplication by the degree-P derivative polynomial cannot reach the inverse core. Its output satisfies |k|>=N+1. The complementary-column bound is

\[
Z_{tail}=|1-a_{tail}\bar\omega/\omega_{ref}|+
\frac{|a_{tail}|}{\omega_{ref}}
\left(\frac1{M+1}+\frac{2\|g'\|_\nu}{N+1}\right).
\]

The identity defect term is retained even when it is numerically negligible. This estimate covers every remaining input index to infinity, not a larger truncation. Let e_G,e_G' be the complete nonlinear errors and Jnorm=max(1,1/omega_ref). The additional all-column error is bounded by

\[
Z_{err}=\|A\|\max\{2\|J\|e_{G'},\;2\tau e_G/\omega_{ref}\}.
\]

Z0=max(Zfinite,Ztail)+Zerr. A separate interval bound on the square finite inverse defect is <1, proving invertibility of the finite block and hence injectivity of A. All weighted induced l1 norms divide by input weights and multiply by output weights.

Y is the interval norm of the complete retained AF(bar z), plus 2||A||||J||e_G for its nonlinear remainder. The phase residual, every retained frequency-dependent exponential, and all finite matrix multiplications are evaluated using intervals.

## Frequency variation and contraction

Write a_J=||A(0,J .)|| and a_P=||A(0,P .)||. These are computed directly, including their tails, rather than replaced by ||A||||J||. Let L0 bound ||G'(bar u)|| and H0=L0||bar u_theta||. With the ball bounds above,

\[
C=\max\left\{
\frac{a_P(1+2\tau L_0)}{\omega_{ref}},\;2a_JW_r,\;
\frac{a_P(1+2\tau L_r)}{\omega_{ref}},\;
\frac{2a_P\tau^2H_0}{\omega_{ref}}
\right\}.
\]

This gives ||A(DF(bar z+v)-DF(bar z))|| <= C||v|| throughout B_r. For the state derivative, use the integrated-shift bound ||J(S_omega-S_baromega)||<=tau |delta omega|/omega_ref on nonzero modes; the zero mode has no shift dependence. For the frequency column, split G(u)-G(bar u) and the shift of G(bar u), whose temporal derivative is bounded by H0. These estimates control both frequency dependence of the delay and the state perturbation.

With T=I-AF, the outward-rounded checks are

\[
Y+(Z_0-1)r+\tfrac12Cr^2<0,\qquad Z_0+Cr<1.
\]

Taylor integration gives a self-map, Banach's theorem gives its unique fixed point, and injectivity of A makes this a zero of F. The solution is real, analytic on the interior Fourier strip, and satisfies the original DDE with the fixed parameters. Uniqueness is in the stated ball and finite phase slice. A nonzero temporal harmonic excludes an equilibrium; this does not assert orbital stability, chaos, or minimality of the enclosed period.

## Executed result and checks

At exact decimal r=0.0006:

- Y <= 4.194761221e-4; Z0 <= 0.069980673; C <= 494.208949.
- Radii polynomial <= -4.957786349e-5; contraction <= 0.366506043.
- Maximum reciprocal defect <= 3.90224e-14; max B_j r <= 0.003483453.
- Physical period is enclosed in [46.7101634894598, 46.751870759002166] (binary64 endpoints in the JSON; the paper rounds outward further).
- A nonconstant Fourier coefficient has modulus >0.14895 throughout the ball.
- Zero true-system Newton steps; zero change to the source waveform and frequency.

The interval proof was executed in MATLAB R2024b with INTLAB 13. `cap.log` records the run. Four Python checks verify the independent finite-difference Jacobian (including frequency-dependent delay), partial-fraction identity, exact source identity, and certificate consistency. They are cross-checks, not substitutes for INTLAB. Separate executed negative tests reject r=1e-12 (self-map inequality fails) and r=0.2 (pole-neighborhood condition fails), with no success certificate written; see `guard_tests.log`.

This proof supersedes the archived free-damping calculation in `proof_MG_zero_correction_candidate.m`. Its old reported radii are not used. The fixed-parameter proof is independent of that driver's nonlinear and radii-polynomial routines.
