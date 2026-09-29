"""Numerical utilities for ordinary periodic orbits of Neumann KS.

The primary unknown used by the integrator is the velocity ``u = h_x`` on
``[-ell, ell]``.  It is restricted to odd functions and hence has a sine
series.  Restriction to ``[0, ell]`` gives

    u_t = -u u_x - u_xx - u_xxxx,
    u(0,t) = u(ell,t) = u_xx(0,t) = u_xx(ell,t) = 0.

Equivalently, the potential h is even and satisfies the homogeneous Neumann
conditions h_x = h_xxx = 0.  The odd subspace is invariant, unlike the
cosine subspace obtained by imposing u_x = u_xxx = 0 directly on velocity.

Periodic orbits are represented by velocity sine coefficients sampled at an
odd number of equally spaced temporal phases.  This makes the temporal
Fourier differentiation matrix exact on the retained trigonometric space and
keeps the finite Newton system square after adding the unknown frequency and
one temporal phase condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import circulant
from scipy.optimize import NoConvergence, least_squares, newton_krylov
from scipy.sparse.linalg import LinearOperator
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NeumannKSConfig:
    interval_length: float = 22.0
    spatial_modes: int = 20
    full_grid_nodes: int = 128
    internal_dt: float = 0.025
    output_dt: float = 0.1
    burn_time: float = 300.0
    sample_time: float = 1600.0
    temporal_modes: int = 28
    random_seed: int = 7
    surrogate: object = field(default=None, compare=False, hash=False, repr=False)

    @property
    def extension_length(self) -> float:
        return 2.0 * self.interval_length


@dataclass
class OrdinaryOrbit:
    frequency: float
    time_values: np.ndarray
    reference_values: np.ndarray
    residual_l2: float
    residual_linf: float
    relative_correction: float
    phase_defect: float
    optimizer_success: bool
    nfev: int
    message: str

    @property
    def period(self) -> float:
        return float(2.0 * np.pi / self.frequency)


def spatial_wavenumbers(config: NeumannKSConfig) -> np.ndarray:
    """Sine wavenumbers k*pi/ell for k=1,...,K."""
    return (
        np.pi
        * np.arange(1, config.spatial_modes + 1, dtype=float)
        / config.interval_length
    )


def temporal_differentiation_matrix(node_count: int) -> np.ndarray:
    """Fourier matrix for d/dtheta on an odd equispaced grid."""
    if node_count % 2 != 1:
        raise ValueError("The temporal node count must be odd.")
    offsets = np.arange(node_count, dtype=float)
    column = np.zeros(node_count, dtype=float)
    nonzero = offsets != 0
    column[nonzero] = 0.5 * (-1.0) ** offsets[nonzero] / np.sin(
        np.pi * offsets[nonzero] / node_count
    )
    # circulant(column)[p,q] = column[(p-q) mod N].
    return circulant(column)


@lru_cache(maxsize=None)
def _spatial_operators(config: NeumannKSConfig):
    """Return synthesis, derivative, and sine-projection matrices."""
    node_count = config.full_grid_nodes
    if node_count < 4 * config.spatial_modes:
        raise ValueError(
            "full_grid_nodes must be at least four times spatial_modes "
            "for alias-free quadratic products."
        )
    x = np.arange(node_count, dtype=float) * config.extension_length / node_count
    kappa = spatial_wavenumbers(config)
    sine = np.sin(x[:, None] * kappa[None, :])
    cosine_derivative = np.cos(x[:, None] * kappa[None, :]) * kappa[None, :]
    projection = (2.0 / node_count) * sine.T
    return x, kappa, sine, cosine_derivative, projection


def nonlinear_sine_coefficients(
    coefficients: np.ndarray, config: NeumannKSConfig
) -> np.ndarray:
    """Return sine coefficients of u*u_x for one or many states."""
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=float))
    if config.surrogate is not None:
        kappa = spatial_wavenumbers(config)
        result = (kappa**2-kappa**4)*coefficients - config.surrogate.predict(coefficients)
        return result[0] if len(coefficients) == 1 else result
    _, _, sine, cosine_derivative, projection = _spatial_operators(config)
    values = coefficients @ sine.T
    derivatives = coefficients @ cosine_derivative.T
    result = (projection @ (values * derivatives).T).T
    return result[0] if len(coefficients) == 1 else result


def nonlinear_sine_jacobian(
    coefficients: np.ndarray, config: NeumannKSConfig
) -> np.ndarray:
    """Jacobian of the sine coefficients of u*u_x."""
    coefficients = np.asarray(coefficients, dtype=float)
    if config.surrogate is not None:
        kappa = spatial_wavenumbers(config)
        return np.diag(kappa**2-kappa**4) - config.surrogate.jacobian(coefficients)
    _, _, sine, cosine_derivative, projection = _spatial_operators(config)
    values = sine @ coefficients
    derivatives = cosine_derivative @ coefficients
    return projection @ (
        derivatives[:, None] * sine + values[:, None] * cosine_derivative
    )


def velocity_rhs_sine(
    coefficients: np.ndarray, config: NeumannKSConfig
) -> np.ndarray:
    """Galerkin RHS for the standard velocity-form KS equation."""
    kappa = spatial_wavenumbers(config)
    linear = kappa**2 - kappa**4
    return linear * np.asarray(coefficients) - nonlinear_sine_coefficients(
        coefficients, config
    )


def potential_coefficients_from_velocity(
    velocity_coefficients: np.ndarray, config: NeumannKSConfig
) -> np.ndarray:
    """Map u sine coefficients to mean-zero h cosine coefficients, u=h_x."""
    return -np.asarray(velocity_coefficients) / spatial_wavenumbers(config)


def velocity_coefficients_from_potential(
    potential_coefficients: np.ndarray, config: NeumannKSConfig
) -> np.ndarray:
    """Map mean-zero h cosine coefficients to u=h_x sine coefficients."""
    return -np.asarray(potential_coefficients) * spatial_wavenumbers(config)


def reconstruct_velocity(
    coefficients: np.ndarray,
    points: np.ndarray,
    config: NeumannKSConfig,
) -> np.ndarray:
    """Evaluate a sine series at points in the odd extension."""
    points = np.asarray(points, dtype=float)
    basis = np.sin(points[:, None] * spatial_wavenumbers(config)[None, :])
    return np.asarray(coefficients) @ basis.T


def reconstruct_potential(
    coefficients: np.ndarray,
    points: np.ndarray,
    config: NeumannKSConfig,
) -> np.ndarray:
    """Evaluate the zero-mean cosine potential corresponding to velocity."""
    points = np.asarray(points, dtype=float)
    potential = potential_coefficients_from_velocity(coefficients, config)
    basis = np.cos(points[:, None] * spatial_wavenumbers(config)[None, :])
    return np.asarray(potential) @ basis.T


def boundary_defects(
    coefficients: np.ndarray, config: NeumannKSConfig
) -> dict[str, float]:
    """Evaluate endpoint defects for velocity and potential boundary data."""
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=float))
    kappa = spatial_wavenumbers(config)
    endpoints = np.asarray([0.0, config.interval_length])
    sine = np.sin(endpoints[:, None] * kappa[None, :])
    velocity = coefficients @ sine.T
    velocity_xx = -(coefficients * kappa**2) @ sine.T
    potential_x = velocity
    potential_xxx = velocity_xx
    return {
        "velocity_value": float(np.max(np.abs(velocity))),
        "velocity_second_derivative": float(np.max(np.abs(velocity_xx))),
        "potential_first_derivative": float(np.max(np.abs(potential_x))),
        "potential_third_derivative": float(np.max(np.abs(potential_xxx))),
    }


def _etdrk4_coefficients(linear: np.ndarray, dt: float):
    roots = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    lr = dt * linear[:, None] + roots[None, :]
    e = np.exp(dt * linear)
    e2 = np.exp(0.5 * dt * linear)
    q = dt * np.real(np.mean((np.exp(0.5 * lr) - 1.0) / lr, axis=1))
    f1 = dt * np.real(
        np.mean(
            (-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2))
            / lr**3,
            axis=1,
        )
    )
    f2 = dt * np.real(
        np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1)
    )
    f3 = dt * np.real(
        np.mean(
            (-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr))
            / lr**3,
            axis=1,
        )
    )
    return e, e2, q, f1, f2, f3


def simulate_velocity_sine(
    initial_coefficients: np.ndarray,
    duration: float,
    config: NeumannKSConfig,
    save_dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the odd velocity Galerkin system with ETDRK4."""
    save_dt = config.output_dt if save_dt is None else float(save_dt)
    ratio = save_dt / config.internal_dt
    save_every = int(round(ratio))
    if not np.isclose(save_every * config.internal_dt, save_dt):
        raise ValueError("save_dt must be an integer multiple of internal_dt.")
    steps = int(round(duration / config.internal_dt))
    if not np.isclose(steps * config.internal_dt, duration):
        raise ValueError("duration must be an integer multiple of internal_dt.")

    kappa = spatial_wavenumbers(config)
    linear = kappa**2 - kappa**4
    e, e2, q, f1, f2, f3 = _etdrk4_coefficients(
        linear, config.internal_dt
    )

    def nonlinear(state):
        return -nonlinear_sine_coefficients(state, config)

    state = np.asarray(initial_coefficients, dtype=float).copy()
    saved = np.empty((steps // save_every + 1, config.spatial_modes))
    saved[0] = state
    saved_index = 1
    for step in range(1, steps + 1):
        nv = nonlinear(state)
        a = e2 * state + q * nv
        na = nonlinear(a)
        b = e2 * state + q * na
        nb = nonlinear(b)
        c = e2 * a + q * (2.0 * nb - nv)
        nc = nonlinear(c)
        state = e * state + f1 * nv + 2.0 * f2 * (na + nb) + f3 * nc
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Neumann KS integration produced non-finite values.")
        if step % save_every == 0:
            saved[saved_index] = state
            saved_index += 1
    times = np.arange(len(saved), dtype=float) * save_dt
    return times, saved


def default_initial_coefficients(config: NeumannKSConfig) -> np.ndarray:
    """Deterministic odd initial state with a small seeded perturbation."""
    rng = np.random.default_rng(config.random_seed)
    coefficients = np.zeros(config.spatial_modes)
    coefficients[:5] = [1.2, -0.5, 0.15, 0.0, 0.08]
    coefficients += 1e-3 * rng.standard_normal(config.spatial_modes)
    coefficients[8:] *= 0.1
    return coefficients


def generate_attractor_trajectory(
    config: NeumannKSConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Burn in and return a trajectory used only for ordinary-return seeds."""
    total = config.burn_time + config.sample_time
    times, values = simulate_velocity_sine(
        default_initial_coefficients(config), total, config
    )
    burn_index = int(round(config.burn_time / config.output_dt))
    return times[burn_index:] - times[burn_index], values[burn_index:]


def scan_ordinary_recurrences(
    trajectory: np.ndarray,
    dt: float,
    min_period: float = 20.0,
    max_period: float = 120.0,
    search_stride: int = 2,
    neighbors: int = 100,
    max_candidates: int = 30,
) -> pd.DataFrame:
    """Rank close returns without any spatial alignment or quotient."""
    trajectory = np.asarray(trajectory, dtype=float)
    sampled_indices = np.arange(0, len(trajectory), search_stride)
    sampled = trajectory[sampled_indices]
    scale = np.std(sampled, axis=0)
    scale[scale < 1e-10] = 1.0
    standardized = (sampled - np.mean(sampled, axis=0)) / scale
    tree = cKDTree(standardized)
    count = min(neighbors + 1, len(sampled))
    distances, indices = tree.query(standardized, k=count, workers=-1)
    minimum_lag = int(round(min_period / dt))
    maximum_lag = int(round(max_period / dt))
    raw = []
    for local_start, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):
        start = int(sampled_indices[local_start])
        for distance, local_end in zip(row_distances[1:], row_indices[1:]):
            end = int(sampled_indices[int(local_end)])
            lag = end - start
            if end > start and minimum_lag <= lag <= maximum_lag:
                raw.append((float(distance), start, end))
    raw.sort()

    rows = []
    for standardized_distance, start, end in raw:
        period = (end - start) * dt
        duplicate = any(
            abs(start - old["start_index"]) <= int(round(2.0 / dt))
            and abs(period - old["period"]) <= 2.0
            for old in rows
        )
        if duplicate:
            continue
        segment = trajectory[start : end + 1]
        error = float(np.linalg.norm(trajectory[end] - trajectory[start]))
        diameter = float(np.max(np.linalg.norm(segment - segment[0], axis=1)))
        rows.append(
            {
                "start_index": start,
                "end_index": end,
                "period": float(period),
                "standardized_distance": standardized_distance,
                "return_error": error,
                "segment_diameter": diameter,
                "relative_return_error": error / (diameter + 1e-15),
            }
        )
        if len(rows) >= max_candidates:
            break
    return pd.DataFrame(rows).sort_values(
        ["relative_return_error", "standardized_distance"], ignore_index=True
    )


def resample_segment_to_phase_grid(
    segment: np.ndarray, temporal_modes: int
) -> np.ndarray:
    """Periodically Fourier-resample a seed segment to 2M+1 phases."""
    segment = np.asarray(segment, dtype=float)
    # Drop the approximate duplicate endpoint before periodic interpolation.
    source = segment[:-1]
    source_count = len(source)
    target_count = 2 * temporal_modes + 1
    source_phase = np.arange(source_count, dtype=float) / source_count
    target_phase = np.arange(target_count, dtype=float) / target_count
    harmonics = np.fft.fftfreq(source_count, d=1.0 / source_count)
    coefficients = np.fft.fft(source, axis=0) / source_count
    values = np.exp(
        2j * np.pi * target_phase[:, None] * harmonics[None, :]
    ) @ coefficients
    return values.real


def collocation_residual(
    time_values: np.ndarray,
    frequency: float,
    config: NeumannKSConfig,
) -> np.ndarray:
    """Unpreconditioned velocity KS residual on the space-time grid."""
    time_values = np.asarray(time_values, dtype=float)
    differentiation = temporal_differentiation_matrix(len(time_values))
    kappa = spatial_wavenumbers(config)
    diagonal = kappa**4 - kappa**2
    nonlinear = nonlinear_sine_coefficients(time_values, config)
    return frequency * (differentiation @ time_values) + diagonal * time_values + nonlinear


def oversampled_residual(
    time_values: np.ndarray,
    frequency: float,
    config: NeumannKSConfig,
    oversampling: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the invariance residual away from collocation nodes.

    The retained time-Fourier polynomial is evaluated on a finer phase grid,
    so quadratic temporal aliasing is included in the returned residual.
    """
    coefficients = time_fourier_coefficients(time_values)
    temporal_modes = (len(coefficients) - 1) // 2
    indices = np.arange(-temporal_modes, temporal_modes + 1)
    node_count = oversampling * len(coefficients) + 1
    phases = 2.0 * np.pi * np.arange(node_count, dtype=float) / node_count
    evaluation = np.exp(1j * phases[:, None] * indices[None, :])
    values = (evaluation @ coefficients).real
    derivatives = (evaluation @ (1j * indices[:, None] * coefficients)).real
    kappa = spatial_wavenumbers(config)
    residual = (
        frequency * derivatives
        + (kappa**4 - kappa**2) * values
        + nonlinear_sine_coefficients(values, config)
    )
    return phases, residual


def independent_flow_closure(
    time_values: np.ndarray,
    period: float,
    config: NeumannKSConfig,
    validation_spatial_modes: int | None = None,
    steps_per_period: int = 40000,
) -> dict[str, float]:
    """Check one-period closure with an independent fine ETDRK4 evolution.

    This is a numerical diagnostic, not a rigorous enclosure.  It is useful
    for unstable orbits because it detects a candidate that only solves an
    under-resolved collocation system.
    """
    validation_spatial_modes = (
        max(config.spatial_modes + 12, 48)
        if validation_spatial_modes is None
        else int(validation_spatial_modes)
    )
    initial = np.zeros(validation_spatial_modes)
    initial[: config.spatial_modes] = np.asarray(time_values)[0]
    dt = float(period) / steps_per_period
    validation_config = NeumannKSConfig(
        interval_length=config.interval_length,
        spatial_modes=validation_spatial_modes,
        full_grid_nodes=max(4 * validation_spatial_modes, 192),
        internal_dt=dt,
        output_dt=float(period),
        temporal_modes=config.temporal_modes,
        random_seed=config.random_seed,
    )
    _, trajectory = simulate_velocity_sine(
        initial, float(period), validation_config, save_dt=float(period)
    )
    difference = trajectory[-1] - initial
    scale = max(float(np.linalg.norm(initial)), 1e-15)
    return {
        "independent_closure_l2": float(np.linalg.norm(difference)),
        "independent_closure_linf": float(np.max(np.abs(difference))),
        "independent_relative_closure": float(np.linalg.norm(difference) / scale),
        "validation_spatial_modes": validation_spatial_modes,
        "validation_steps_per_period": int(steps_per_period),
    }


def _phase_condition(
    values: np.ndarray,
    reference: np.ndarray,
    reference_tangent: np.ndarray,
) -> float:
    return float(np.sum((values - reference) * reference_tangent))


def refine_ordinary_orbit(
    seed_values: np.ndarray,
    period_guess: float,
    config: NeumannKSConfig,
    max_nfev: int = 80,
) -> OrdinaryOrbit:
    """Refine a full ordinary return with one temporal phase condition."""
    reference = np.asarray(seed_values, dtype=float)
    node_count, mode_count = reference.shape
    if mode_count != config.spatial_modes:
        raise ValueError("seed_values has the wrong spatial mode count.")
    differentiation = temporal_differentiation_matrix(node_count)
    reference_tangent = differentiation @ reference
    residual_scale = max(
        float(np.sqrt(np.mean(collocation_residual(
            reference, 2.0 * np.pi / period_guess, config
        ) ** 2))),
        1e-6,
    )
    phase_scale = max(float(np.sum(reference_tangent**2)), 1e-10)
    initial = np.r_[reference.ravel(), np.log(2.0 * np.pi / period_guess)]
    lower = np.full_like(initial, -np.inf)
    upper = np.full_like(initial, np.inf)
    frequency_guess = 2.0 * np.pi / period_guess
    lower[-1] = np.log(frequency_guess / 1.3)
    upper[-1] = np.log(frequency_guess / 0.7)

    def unpack(parameters):
        values = parameters[:-1].reshape(node_count, mode_count)
        frequency = float(np.exp(parameters[-1]))
        return values, frequency

    def residual(parameters):
        values, frequency = unpack(parameters)
        equation = collocation_residual(values, frequency, config) / residual_scale
        phase = _phase_condition(values, reference, reference_tangent) / phase_scale
        return np.r_[equation.ravel(), phase]

    def jacobian(parameters):
        values, frequency = unpack(parameters)
        total = node_count * mode_count
        matrix = np.zeros((total + 1, total + 1))
        kappa = spatial_wavenumbers(config)
        diagonal = kappa**4 - kappa**2
        # Flattening is time-major.  D kron I acts across temporal nodes.
        matrix[:total, :total] = frequency * np.kron(
            differentiation, np.eye(mode_count)
        )
        for temporal_index, state in enumerate(values):
            block = np.diag(diagonal) + nonlinear_sine_jacobian(state, config)
            start = temporal_index * mode_count
            matrix[start : start + mode_count, start : start + mode_count] += block
        matrix[:total, -1] = (frequency * (differentiation @ values)).ravel()
        matrix[-1, :total] = reference_tangent.ravel() / phase_scale
        matrix[:total, :] /= residual_scale
        return matrix

    result = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(lower, upper),
        xtol=2e-13,
        ftol=2e-13,
        gtol=2e-13,
        max_nfev=max_nfev,
        x_scale="jac",
        verbose=0,
    )
    values, frequency = unpack(result.x)
    equation = collocation_residual(values, frequency, config)
    correction = np.linalg.norm(values - reference) / max(
        np.linalg.norm(reference - np.mean(reference, axis=0)), 1e-15
    )
    return OrdinaryOrbit(
        frequency=frequency,
        time_values=values,
        reference_values=reference,
        residual_l2=float(np.sqrt(np.mean(equation**2))),
        residual_linf=float(np.max(np.abs(equation))),
        relative_correction=float(correction),
        phase_defect=_phase_condition(values, reference, reference_tangent)
        / phase_scale,
        optimizer_success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
    )


def refine_ordinary_orbit_matrix_free(
    seed_values: np.ndarray,
    frequency_guess: float,
    config: NeumannKSConfig,
    max_nfev: int = 45,
    tight: bool = False,
) -> OrdinaryOrbit:
    """Continue a resolved orbit with a matrix-free preconditioned solve.

    Dense Jacobians become impractical beyond a few thousand space--time
    coefficients.  This implementation applies both the Jacobian and its
    transpose as linear operators.  The equation residual is preconditioned
    mode-by-mode by the inverse linear KS symbol

        (i*j*omega + kappa_k^4-kappa_k^2)^(-1).

    The phase equation remains the same single temporal phase condition.
    """
    reference = np.asarray(seed_values, dtype=float)
    node_count, mode_count = reference.shape
    if node_count != 2 * config.temporal_modes + 1:
        raise ValueError("seed_values has the wrong temporal node count.")
    if mode_count != config.spatial_modes:
        raise ValueError("seed_values has the wrong spatial mode count.")

    differentiation = temporal_differentiation_matrix(node_count)
    reference_tangent = differentiation @ reference
    phase_scale = max(float(np.sum(reference_tangent**2)), 1e-15)
    time_indices = np.arange(
        -config.temporal_modes, config.temporal_modes + 1
    )[:, None]
    kappa = spatial_wavenumbers(config)[None, :]
    linear = kappa**4 - kappa**2
    _, _, sine, cosine_derivative, projection = _spatial_operators(config)

    def apply_linear_inverse(values, frequency, adjoint=False):
        coefficients = np.fft.fftshift(
            np.fft.fft(values, axis=0) / node_count, axes=0
        )
        denominator = 1j * frequency * time_indices + linear
        if adjoint:
            denominator = np.conj(denominator)
        return time_values_from_fourier(coefficients / denominator)

    def inverse_frequency_derivative(values, frequency):
        coefficients = np.fft.fftshift(
            np.fft.fft(values, axis=0) / node_count, axes=0
        )
        denominator = 1j * frequency * time_indices + linear
        multiplier = -1j * frequency * time_indices / denominator**2
        return time_values_from_fourier(coefficients * multiplier)

    def nonlinear_derivative(values, directions):
        if config.surrogate is not None:
            return -linear*directions - np.einsum(
                'tij,tj->ti', config.surrogate.jacobian(values), directions)
        value_nodes = values @ sine.T
        derivative_nodes = values @ cosine_derivative.T
        direction_nodes = directions @ sine.T
        direction_derivatives = directions @ cosine_derivative.T
        return (
            projection
            @ (
                direction_nodes * derivative_nodes
                + value_nodes * direction_derivatives
            ).T
        ).T

    def nonlinear_derivative_transpose(values, dual_values):
        if config.surrogate is not None:
            return -linear*dual_values - np.einsum(
                'tij,ti->tj', config.surrogate.jacobian(values), dual_values)
        value_nodes = values @ sine.T
        derivative_nodes = values @ cosine_derivative.T
        projected_dual = dual_values @ projection
        return (
            (projected_dual * derivative_nodes) @ sine
            + (projected_dual * value_nodes) @ cosine_derivative
        )

    def unpack(parameters):
        values = parameters[:-1].reshape(node_count, mode_count)
        return values, float(np.exp(parameters[-1]))

    def residual(parameters):
        values, frequency = unpack(parameters)
        equation = collocation_residual(values, frequency, config)
        preconditioned = apply_linear_inverse(equation, frequency)
        phase = _phase_condition(
            values, reference, reference_tangent
        ) / phase_scale
        return np.r_[preconditioned.ravel(), phase]

    def jacobian(parameters):
        values, frequency = unpack(parameters)
        equation = collocation_residual(values, frequency, config)
        time_derivative = differentiation @ values
        inverse_frequency_term = inverse_frequency_derivative(
            equation, frequency
        )
        dimension = node_count * mode_count + 1

        def matvec(direction):
            direction_values = direction[:-1].reshape(node_count, mode_count)
            direction_log_frequency = direction[-1]
            equation_direction = (
                frequency * (differentiation @ direction_values)
                + linear * direction_values
                + nonlinear_derivative(values, direction_values)
                + direction_log_frequency * frequency * time_derivative
            )
            preconditioned_direction = apply_linear_inverse(
                equation_direction, frequency
            ) + direction_log_frequency * inverse_frequency_term
            phase_direction = (
                np.sum(direction_values * reference_tangent) / phase_scale
            )
            return np.r_[preconditioned_direction.ravel(), phase_direction]

        def rmatvec(dual):
            equation_dual = dual[:-1].reshape(node_count, mode_count)
            phase_dual = dual[-1]
            pulled_back = apply_linear_inverse(
                equation_dual, frequency, adjoint=True
            )
            values_gradient = (
                frequency * (differentiation.T @ pulled_back)
                + linear * pulled_back
                + nonlinear_derivative_transpose(values, pulled_back)
                + phase_dual * reference_tangent / phase_scale
            )
            frequency_gradient = (
                np.sum(pulled_back * frequency * time_derivative)
                + np.sum(equation_dual * inverse_frequency_term)
            )
            return np.r_[values_gradient.ravel(), frequency_gradient]

        return LinearOperator(
            (dimension, dimension),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=float,
        )

    initial = np.r_[reference.ravel(), np.log(float(frequency_guess))]
    # A few Jacobian-free Newton--Krylov steps cheaply remove the newly
    # resolved Fourier tail before the transpose-aware least-squares solve.
    try:
        warmed_initial = newton_krylov(
            residual,
            initial,
            method="lgmres",
            maxiter=25,
            f_tol=1e-7,
            inner_maxiter=80,
        )
    except NoConvergence as error:
        warmed_initial = np.asarray(error.args[0], dtype=float)
    solver_options = {
        "jac": jacobian,
        "method": "trf",
        "tr_solver": "lsmr",
        "max_nfev": max_nfev,
        "xtol": 1e-12,
        "ftol": 1e-12,
        "gtol": 1e-13 if tight else 1e-10,
        "verbose": 0,
    }
    if tight:
        solver_options["tr_options"] = {
            "atol": 1e-14,
            "btol": 1e-14,
            "maxiter": 500,
        }
    result = least_squares(residual, warmed_initial, **solver_options)
    values, frequency = unpack(result.x)
    equation = collocation_residual(values, frequency, config)
    preconditioned_linf = float(np.max(np.abs(residual(result.x))))
    correction = np.linalg.norm(values - reference) / max(
        np.linalg.norm(reference - np.mean(reference, axis=0)), 1e-15
    )
    return OrdinaryOrbit(
        frequency=frequency,
        time_values=values,
        reference_values=reference,
        residual_l2=float(np.sqrt(np.mean(equation**2))),
        residual_linf=float(np.max(np.abs(equation))),
        relative_correction=float(correction),
        phase_defect=_phase_condition(
            values, reference, reference_tangent
        ) / phase_scale,
        optimizer_success=bool(
            result.success
            or (
                preconditioned_linf < 5e-6
                and np.sqrt(np.mean(equation**2)) < 1e-5
            )
        ),
        nfev=int(result.nfev),
        message=(
            f"{result.message} preconditioned_linf={preconditioned_linf:.3e}"
        ),
    )


def time_fourier_coefficients(time_values: np.ndarray) -> np.ndarray:
    """Return centered complex time-Fourier coefficients a[j,k]."""
    values = np.asarray(time_values, dtype=float)
    return np.fft.fftshift(np.fft.fft(values, axis=0) / len(values), axes=0)


def time_values_from_fourier(coefficients: np.ndarray) -> np.ndarray:
    """Inverse of time_fourier_coefficients on the retained grid."""
    coefficients = np.asarray(coefficients, dtype=complex)
    values = np.fft.ifft(
        np.fft.ifftshift(coefficients, axes=0) * len(coefficients), axis=0
    )
    return values.real


def save_candidate(
    path: str | Path,
    orbit: OrdinaryOrbit,
    config: NeumannKSConfig,
    newton_corrections: int = 0,
) -> None:
    """Save a frozen CAP input; no correction is performed here."""
    coefficients = time_fourier_coefficients(orbit.time_values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frequency=orbit.frequency,
        period=orbit.period,
        velocity_time_fourier=coefficients,
        velocity_time_values=orbit.time_values,
        reference_time_values=orbit.reference_values,
        interval_length=config.interval_length,
        extension_length=config.extension_length,
        spatial_modes=config.spatial_modes,
        temporal_modes=config.temporal_modes,
        full_grid_nodes=config.full_grid_nodes,
        newton_corrections=int(newton_corrections),
        residual_l2=orbit.residual_l2,
        residual_linf=orbit.residual_linf,
        phase_defect=orbit.phase_defect,
        relative_correction=orbit.relative_correction,
    )


def load_candidate(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def candidate_summary(
    orbit: OrdinaryOrbit,
    config: NeumannKSConfig,
    source_period: float,
    source_return_error: float,
    validation: dict[str, float] | None = None,
) -> pd.DataFrame:
    defects = boundary_defects(orbit.time_values, config)
    return pd.DataFrame(
        [
            {
                "formulation": "potential Neumann / odd velocity",
                "orbit_type": "ordinary periodic",
                "interval_length": config.interval_length,
                "odd_extension_length": config.extension_length,
                "spatial_modes": config.spatial_modes,
                "temporal_modes": config.temporal_modes,
                "seed_period": source_period,
                "seed_relative_return_error": source_return_error,
                "refined_period": orbit.period,
                "collocation_residual_l2": orbit.residual_l2,
                "collocation_residual_linf": orbit.residual_linf,
                "relative_seed_to_orbit_correction": orbit.relative_correction,
                "temporal_phase_defect": orbit.phase_defect,
                "optimizer_success": orbit.optimizer_success,
                "optimizer_function_evaluations": orbit.nfev,
                "cap_transfer_newton_corrections": 0,
                **defects,
                **({} if validation is None else validation),
            }
        ]
    )
