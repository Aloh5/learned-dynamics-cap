# Newton-refined Neumann KS certificate

This workflow is separate from `outputs_surrogate/` and preserves the
zero-correction result. It corrects the original Neumann KS equation, not the
periodic-domain forecasting problem.

## Candidate and fixed operator

`newton_refine.py` embeds the projected surrogate export in 220 temporal and
100 spatial modes, fixes its original temporal phase condition, and performs
three Newton--GMRES updates. The equation is evaluated by fully de-aliased
Fourier products. The original 160-time/40-space finite inverse B and original
frequency tail symbols are used as a fixed preconditioner throughout.
The spatial interval remains length 22 and all physical parameters are fixed.
Numerical residuals are diagnostics, not existence proofs.

## Infinite-dimensional proof

The original executed certificate and its Fourier proof-data file define a
fixed preconditioned operator H=A F, including its phase equation. Neither A
nor the smoothing operator is recentered. Because F is quadratic in the
frequency and waveform, its quadratic remainder estimate is global in the
same Banach space. If d bounds the new center's distance from the original
center, the original bounds imply

    alpha_new <= alpha_original + 2 Z2_original d,
    Z2_new = Z2_original.

These inequalities cover all infinite modes. They do not treat the old scan
as if it had been regenerated at the corrected center. The unchanged phase
row has no derivative variation. The original source coefficients and
frequency are checked against the saved proof data.

`proof_KS_newton_refined.m` evaluates d, the new residual Y, and the final
radii inequalities with INTLAB. The full quadratic residual is evaluated by
**direct interval coefficient convolution**, not quadrature or FFT. For each
pair of spatial sine modes j,k, temporal convolution is computed by an
interval Toeplitz matrix-vector product. Exact sine product identities give
half the derivative of the square. All temporal modes through 440 and all
spatial modes through 200 are retained. The real sine/cosine conversion and
invariant n+k parity are explicit. This covers the complete polynomial
residual; all higher coefficients are exactly zero at the finite center.
The original infinite-space derivative bounds cover corrections outside it.

The finite residual vector is multiplied by interval B directly. Its vector
ordering is the same temporal-major, spatial-minor real-basis ordering as in
the original scan. The complementary residual uses the original raw linear
inverse. Phase residual and frequency uncertainty are included. The proof
checks a nonzero temporal coefficient, positivity of frequency, and interval
containment of the entire refined ball inside the original 1e-5 ball. Since
the phase slice and equation agree, original local uniqueness identifies the
same periodic solution.

This sharp residual calculation and the enlarged candidate truncation both
contribute to accuracy. Improvement of the certificate must not be attributed
solely to Newton updates: the original certificate used a more conservative
trigonometric-transform residual bound. There is no controlled comparison
here isolating each source of improvement.

## Reproduce

From the repository root:

```sh
PYTHON_BINARY=.venv/bin/python bash CAP/Kuramoto_PO/run_newton_refined_proof.sh
```

Requires the existing original `outputs_surrogate/` proof artifacts, MATLAB,
and INTLAB 13. `MATLAB_BINARY`, `INTLAB_PATH`, and `KS_REFINED_RADIUS` are
configurable. The original proof scan remains part of the dependency chain;
its forward-error audit is in `PROOF_AUDIT.md`.
The results and hashes are generated in `outputs_newton_refined/` only after
successful interval validation. The claim is existence and phase/symmetry
restricted local uniqueness, not orbital stability, chaos, or a minimal period.
