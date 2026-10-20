"""Complete polynomial libraries and dense regression; no support selection.

Coefficients have shape (outputs, features). MG uses RMS-normalized,
degree-weighted ridge; periodic KS uses unscaled minimum-norm least squares.
"""
from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import lstsq


def polynomial_exponents(dimension, degree):
    if dimension < 1 or degree < 0:
        raise ValueError('dimension must be positive and degree nonnegative')
    powers = []
    for order in range(degree + 1):
        for indices in combinations_with_replacement(range(dimension), order):
            powers.append(np.bincount(indices, minlength=dimension))
    return np.asarray(powers, dtype=int)


def polynomial_feature_names(powers):
    return np.asarray([' '.join(f'x{j}' if p == 1 else f'x{j}^{p}'
        for j, p in enumerate(row) if p) or '1' for row in powers], dtype=object)


class PolynomialLibrary:
    def __init__(self, powers):
        self.powers_ = np.asarray(powers, dtype=int)

    def transform(self, states):
        x = np.atleast_2d(np.asarray(states, dtype=float))
        return np.prod(x[:, None, :] ** self.powers_[None, :, :], axis=2)

    def get_feature_names(self, input_features=None):
        return list(polynomial_feature_names(self.powers_))

    def get_feature_names_out(self, input_features=None):
        return polynomial_feature_names(self.powers_)


class LibraryModel:
    """Dense vector field in a supplied feature library (also used by ablations)."""
    def __init__(self, coefficients, library):
        self._coefficients = np.asarray(coefficients, dtype=float)
        self.feature_library = library

    def predict(self, states):
        return self.feature_library.transform(states) @ self._coefficients.T

    def coefficients(self):
        return self._coefficients.copy()

    def print(self, precision=3):
        for j, row in enumerate(self._coefficients):
            terms = [f'{c:+.{precision}f} {name}' for c, name in
                     zip(row, self.feature_library.get_feature_names_out()) if abs(c) > 1e-12]
            print(f'(x{j})\' = ' + ' '.join(terms))


class PolynomialModel(LibraryModel):
    def __init__(self, coefficients, powers, diagnostics=None):
        super().__init__(coefficients, PolynomialLibrary(powers))
        self.diagnostics = diagnostics or {}

    def jacobian(self, state):
        x = np.asarray(state, dtype=float)
        powers = self.feature_library.powers_
        columns = []
        for j in range(len(x)):
            reduced = powers.copy()
            reduced[:, j] = np.maximum(0, reduced[:, j] - 1)
            columns.append(powers[:, j] * np.prod(x ** reduced, axis=1))
        return self._coefficients @ np.column_stack(columns)

    def save(self, path):
        np.savez(path, coefficients=self._coefficients, powers=self.feature_library.powers_)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data['coefficients'], data['powers'])


def solve_dense_regression(features, targets, *, regularization=0.0,
                           normalize=False, weights=None, driver='gelsd'):
    """Solve by SVD or rank-revealing QR, with augmented rows for ridge.

    No normal matrix is formed. Zero regularization and no scaling preserve the
    minimum-Euclidean-norm convention for rank-deficient polynomial libraries.
    Diagnostics describe the training library, before ridge augmentation.
    """
    theta = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if theta.ndim != 2 or y.ndim != 2 or len(theta) != len(y) or not len(theta):
        raise ValueError('features and targets must be nonempty aligned matrices')
    if not np.isfinite(theta).all() or not np.isfinite(y).all():
        raise ValueError('training matrices must be finite')
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError('regularization must be finite and nonnegative')
    scales = np.sqrt(np.mean(theta**2, axis=0)) if normalize else np.ones(theta.shape[1])
    scales[scales < 1e-14] = 1.0
    design = theta / scales
    weights = np.ones(theta.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (theta.shape[1],) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('weights must be finite and nonnegative, one per feature')
    if regularization:
        singular = np.linalg.svd(design, compute_uv=False)
        cutoff = np.finfo(float).eps * singular[0]
        rank = int(np.count_nonzero(singular > cutoff))
        matrix = np.vstack([design, np.diag(np.sqrt(regularization * weights))])
        rhs = np.vstack([y, np.zeros((theta.shape[1], y.shape[1]))])
        fitted = lstsq(matrix, rhs, lapack_driver=driver)[0]
    else:
        fitted, _, rank, singular = lstsq(design, y, lapack_driver=driver)
    coefficients = fitted.T / scales
    condition = (float(singular[0] / singular[-1])
                 if singular is not None and len(singular) and singular[-1] > 0 else None)
    diagnostics = dict(samples=len(theta), features=theta.shape[1], rank=int(rank),
        full_column_rank=bool(rank == theta.shape[1]), condition_number=condition,
        singular_values=None if singular is None else singular.tolist(),
        regularization=float(regularization), normalized=bool(normalize), driver=driver,
        relative_derivative_residual=float(np.linalg.norm(theta @ coefficients.T-y) /
                                          max(np.linalg.norm(y), np.finfo(float).tiny)))
    return coefficients, diagnostics


def fit_polynomial_vector_field(states, derivatives, *, degree=2,
                                regularization=0.0, normalize=False, degree_weight=1.0):
    x = np.asarray(states, dtype=float)
    y = np.asarray(derivatives, dtype=float)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError('states and derivatives must have identical 2D shapes')
    powers = polynomial_exponents(x.shape[1], degree)
    theta = PolynomialLibrary(powers).transform(x)
    weights = np.maximum(powers.sum(axis=1), 1).astype(float) ** degree_weight
    weights[0] = 0.0
    coefficients, diagnostics = solve_dense_regression(theta, y,
        regularization=regularization, normalize=normalize, weights=weights)
    return PolynomialModel(coefficients, powers, diagnostics)
