from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .library import polynomial_powers, transform_polynomial


@dataclass
class PolynomialVectorField:
    coefficients: np.ndarray  # shape: (n_outputs, n_features)
    powers: np.ndarray
    method: str
    hyperparameter: str | float
    ridge_alpha: float = 0.0
    threshold: float = 0.0

    def predict(self, states: np.ndarray) -> np.ndarray:
        return transform_polynomial(states, self.powers) @ self.coefficients.T

    @property
    def active_terms(self) -> int:
        return int(np.count_nonzero(self.coefficients))

    @property
    def total_terms(self) -> int:
        return int(self.coefficients.size)

    @property
    def active_features(self) -> int:
        return int(np.count_nonzero(np.linalg.norm(self.coefficients, axis=0)))


def _library_and_scaling(states: np.ndarray, degree: int):
    powers = polynomial_powers(states.shape[1], degree)
    theta = transform_polynomial(states, powers)
    scales = np.sqrt(np.mean(theta**2, axis=0))
    scales[scales < 1e-14] = 1.0
    return powers, theta / scales, scales


def _ridge(theta: np.ndarray, targets: np.ndarray, alpha: float, penalty: np.ndarray) -> np.ndarray:
    # Solve the (optionally augmented) least-squares problem directly. Forming
    # theta.T @ theta squares the condition number and destabilizes the KS nodal
    # library, whose condition number is of order 1e17.
    if alpha > 0:
        regularizer = np.sqrt(alpha) * np.diag(np.sqrt(penalty))
        design = np.vstack([theta, regularizer])
        response = np.vstack([targets, np.zeros((theta.shape[1], targets.shape[1]))])
    else:
        design, response = theta, targets
    return np.linalg.lstsq(design, response, rcond=None)[0]


def fit_ddsr(
    states: np.ndarray,
    derivatives: np.ndarray,
    degree: int,
    alpha: float,
    degree_weight: float = 1.0,
) -> PolynomialVectorField:
    """Fit the full RMS-normalized, degree-weighted ridge library used by DDSR."""
    states = np.asarray(states, dtype=float)
    derivatives = np.asarray(derivatives, dtype=float)
    powers, theta_scaled, scales = _library_and_scaling(states, degree)
    penalty = np.maximum(powers.sum(axis=1), 1).astype(float) ** degree_weight
    penalty[0] = 0.0
    coefficients_scaled = _ridge(theta_scaled, derivatives, alpha, penalty)
    coefficients = (coefficients_scaled / scales[:, None]).T
    return PolynomialVectorField(
        coefficients,
        powers,
        "DDSR",
        f"alpha={alpha:g}",
        ridge_alpha=float(alpha),
        threshold=0.0,
    )


def fit_sindy_stlsq(
    states: np.ndarray,
    derivatives: np.ndarray,
    degree: int,
    threshold: float,
    alpha: float = 0.0,
    max_iter: int = 20,
    normalize_columns: bool = True,
) -> PolynomialVectorField:
    """Standard sequential thresholded least squares (STLSQ).

    Thresholding is performed on RMS-normalized library coefficients by default,
    so a single threshold has a comparable meaning across monomial degrees. Each
    output equation is refit on its surviving support after every threshold step.
    """
    states = np.asarray(states, dtype=float)
    derivatives = np.asarray(derivatives, dtype=float)
    powers = polynomial_powers(states.shape[1], degree)
    theta = transform_polynomial(states, powers)
    scales = np.sqrt(np.mean(theta**2, axis=0)) if normalize_columns else np.ones(theta.shape[1])
    scales[scales < 1e-14] = 1.0
    design = theta / scales
    penalty = np.ones(design.shape[1])
    penalty[0] = 0.0
    initial = _ridge(design, derivatives, alpha, penalty)
    coefficients_scaled = np.zeros_like(initial)

    for output in range(derivatives.shape[1]):
        support = np.ones(design.shape[1], dtype=bool)
        coef = initial[:, output]
        for _ in range(max_iter):
            new_support = np.abs(coef) >= threshold
            if not np.any(new_support):
                new_support[np.argmax(np.abs(coef))] = True
            if np.array_equal(new_support, support):
                break
            support = new_support
            coef = np.zeros_like(coef)
            local_penalty = penalty[support]
            coef[support] = _ridge(
                design[:, support], derivatives[:, [output]], alpha, local_penalty
            )[:, 0]
        coefficients_scaled[:, output] = coef

    coefficients = (coefficients_scaled / scales[:, None]).T
    return PolynomialVectorField(
        coefficients,
        powers,
        "SINDy-STLSQ",
        f"alpha={alpha:g}; threshold={threshold:g}",
        ridge_alpha=float(alpha),
        threshold=float(threshold),
    )


def derivative_nrmse(model: PolynomialVectorField, states: np.ndarray, derivatives: np.ndarray) -> float:
    residual = model.predict(states) - derivatives
    return float(np.sqrt(np.mean(residual**2)) / (np.std(derivatives) + 1e-12))
