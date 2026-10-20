"""Relative-periodic-orbit proposal and validation utilities for the KS notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial import cKDTree


def spectral_shift(state, distance, domain_length):
    """Return u(x + distance) for a periodic nodal state."""
    state = np.asarray(state, dtype=float)
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(
        len(state), d=domain_length / len(state)
    )
    shifted_hat = np.fft.rfft(state) * np.exp(1j * wavenumbers * distance)
    if len(state) % 2 == 0:
        shifted_hat[-1] = shifted_hat[-1].real
    return np.fft.irfft(shifted_hat, n=len(state))


def spatial_tangent(state, domain_length):
    """Infinitesimal generator of spatial translations."""
    state = np.asarray(state, dtype=float)
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(
        len(state), d=domain_length / len(state)
    )
    tangent_hat = 1j * wavenumbers * np.fft.rfft(state)
    if len(state) % 2 == 0:
        tangent_hat[-1] = tangent_hat[-1].real
    return np.fft.irfft(tangent_hat, n=len(state))


def wrap_shift(distance, domain_length):
    """Represent a periodic shift in [-L/2, L/2)."""
    return float(
        (distance + 0.5 * domain_length) % domain_length
        - 0.5 * domain_length
    )


def canonicalize_translation(state, domain_length):
    """First-Fourier-mode slice used only for recurrence detection."""
    state_hat = np.fft.rfft(np.asarray(state, dtype=float))
    if len(state_hat) < 2 or abs(state_hat[1]) < 1e-10:
        return np.asarray(state, dtype=float).copy(), 0.0
    first_wavenumber = 2.0 * np.pi / domain_length
    slice_shift = -np.angle(state_hat[1]) / first_wavenumber
    return (
        spectral_shift(state, slice_shift, domain_length),
        wrap_shift(slice_shift, domain_length),
    )


def best_relative_shift(first, second, domain_length, grid_factor=6):
    """Find ell minimizing ||g(-ell) second - first||."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    grid_count = max(grid_factor * len(first), 32)
    shifts = np.linspace(
        -0.5 * domain_length,
        0.5 * domain_length,
        grid_count,
        endpoint=False,
    )

    def objective(shift):
        return np.linalg.norm(
            spectral_shift(second, -shift, domain_length) - first
        )

    values = np.asarray([objective(shift) for shift in shifts])
    best_index = int(np.argmin(values))
    grid_step = domain_length / grid_count
    center = shifts[best_index]
    refined = minimize_scalar(
        objective,
        bounds=(center - grid_step, center + grid_step),
        method="bounded",
        options={"xatol": 1e-10},
    )
    shift = wrap_shift(refined.x, domain_length)
    return shift, float(objective(shift))


def scan_relative_recurrences(
    trajectory,
    dt,
    domain_length,
    min_period=5.0,
    max_period=80.0,
    search_stride=2,
    neighbors=60,
    max_candidates=30,
):
    """Use a translation-reduced KD tree to find candidate relative returns."""
    trajectory = np.asarray(trajectory, dtype=float)
    sampled_indices = np.arange(0, len(trajectory), search_stride)
    canonical = np.asarray(
        [
            canonicalize_translation(trajectory[index], domain_length)[0]
            for index in sampled_indices
        ]
    )
    scale = np.std(canonical, axis=0)
    scale[scale < 1e-10] = 1.0
    standardized = (canonical - np.mean(canonical, axis=0)) / scale
    tree = cKDTree(standardized)
    query_count = min(neighbors + 1, len(standardized))
    distances, neighbor_indices = tree.query(
        standardized, k=query_count, workers=-1
    )

    min_lag = int(round(min_period / dt))
    max_lag = int(round(max_period / dt))
    raw = []
    for local_start, (row_distances, row_neighbors) in enumerate(
        zip(distances, neighbor_indices)
    ):
        start = int(sampled_indices[local_start])
        for distance, local_end in zip(row_distances[1:], row_neighbors[1:]):
            end = int(sampled_indices[int(local_end)])
            if end <= start:
                continue
            lag = end - start
            if min_lag <= lag <= max_lag:
                raw.append((float(distance), start, end))

    raw.sort(key=lambda item: item[0])
    rows = []
    start_tolerance = max(1, int(round(2.0 / dt)))
    for canonical_distance, start, end in raw:
        period = (end - start) * dt
        duplicate = any(
            abs(start - old["start_index"]) <= start_tolerance
            and abs(period - old["period"]) <= 1.0
            for old in rows
        )
        if duplicate:
            continue

        spatial_shift_value, return_error = best_relative_shift(
            trajectory[start], trajectory[end], domain_length
        )
        segment = trajectory[start : end + 1]
        diameter = float(
            np.max(np.linalg.norm(segment - segment[0], axis=1))
        )
        rows.append(
            {
                "start_index": start,
                "end_index": end,
                "period": float(period),
                "spatial_shift": spatial_shift_value,
                "canonical_distance": canonical_distance,
                "return_error": return_error,
                "segment_diameter": diameter,
                "relative_return_error": float(
                    return_error / (diameter + 1e-12)
                ),
            }
        )
        if len(rows) >= max_candidates:
            break

    if not rows:
        return pd.DataFrame(
            columns=[
                "start_index",
                "end_index",
                "period",
                "spatial_shift",
                "canonical_distance",
                "return_error",
                "segment_diameter",
                "relative_return_error",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["relative_return_error", "canonical_distance"]
    ).reset_index(drop=True)


def rk4_endpoint(model, initial_state, period, n_steps, model_rhs):
    """Fixed-step RK4 endpoint for the learned continuous-time model."""
    dt = period / n_steps
    state = np.asarray(initial_state, dtype=float).copy()
    for _ in range(n_steps):
        k1 = model_rhs(model, state)
        k2 = model_rhs(model, state + 0.5 * dt * k1)
        k3 = model_rhs(model, state + 0.5 * dt * k2)
        k4 = model_rhs(model, state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > 1e4:
            return np.full_like(state, np.nan)
    return state


def adaptive_endpoint(
    initial_state,
    period,
    rhs_function,
    rtol=1e-9,
    atol=1e-11,
    max_step=0.2,
):
    """High-accuracy DOP853 endpoint for an autonomous semidiscrete flow."""
    solution = solve_ivp(
        lambda _time, state: rhs_function(state),
        (0.0, float(period)),
        np.asarray(initial_state, dtype=float),
        method="DOP853",
        t_eval=[float(period)],
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not solution.success or solution.y.shape[1] == 0:
        return np.full_like(np.asarray(initial_state, dtype=float), np.nan)
    return solution.y[:, -1]


def relative_closure_defect(
    state, endpoint, spatial_shift_value, domain_length
):
    """Normalized full-state relative-periodicity defect."""
    state = np.asarray(state, dtype=float)
    endpoint = np.asarray(endpoint, dtype=float)
    aligned = spectral_shift(
        endpoint, -spatial_shift_value, domain_length
    )
    scale = max(np.linalg.norm(state - np.mean(state)), 1e-12)
    return float(np.linalg.norm(aligned - state) / scale)


@dataclass
class OrbitRefinement:
    state: np.ndarray
    period: float
    spatial_shift: float
    endpoint: np.ndarray
    optimizer_success: bool
    converged: bool
    nfev: int
    cost: float
    relative_closure_defect: float
    mean_closure_defect: float
    time_phase_defect: float
    space_phase_defect: float
    relative_state_correction: float
    speed_norm: float
    message: str


def refine_relative_orbit(
    reference_state,
    period_guess,
    shift_guess,
    domain_length,
    endpoint_function,
    rhs_function,
    max_step=0.1,
    max_nfev=80,
    defect_tolerance=1e-7,
    period_window=(0.7, 1.3),
):
    """Single-shoot a relative periodic orbit at fixed spatial mean."""
    reference_state = np.asarray(reference_state, dtype=float)
    n_state = len(reference_state)
    mean_level = float(np.mean(reference_state))
    mean_vector = np.full(n_state, mean_level)
    basis = null_space(np.ones((1, n_state)))
    q_reference = basis.T @ (reference_state - mean_vector)
    time_tangent = np.asarray(rhs_function(reference_state), dtype=float)
    space_tangent_value = spatial_tangent(reference_state, domain_length)
    state_scale = max(
        np.linalg.norm(reference_state - mean_vector), 1e-8
    )
    time_scale = max(
        state_scale * np.linalg.norm(time_tangent), 1e-10
    )
    space_scale = max(
        state_scale * np.linalg.norm(space_tangent_value), 1e-10
    )
    n_steps = max(20, int(np.ceil(period_guess / max_step)))

    initial = np.concatenate(
        [q_reference, [np.log(period_guess), shift_guess]]
    )
    lower = np.full_like(initial, -np.inf)
    upper = np.full_like(initial, np.inf)
    lower[-2] = np.log(period_window[0] * period_guess)
    upper[-2] = np.log(period_window[1] * period_guess)
    lower[-1] = shift_guess - domain_length
    upper[-1] = shift_guess + domain_length

    def unpack(parameters):
        state = mean_vector + basis @ parameters[:-2]
        period = float(np.exp(parameters[-2]))
        spatial_shift_value = float(parameters[-1])
        return state, period, spatial_shift_value

    def residual(parameters):
        state, period, spatial_shift_value = unpack(parameters)
        endpoint = endpoint_function(state, period, n_steps)
        if not np.all(np.isfinite(endpoint)):
            return np.full(n_state + 1, 1e6)
        aligned_endpoint = spectral_shift(
            endpoint, -spatial_shift_value, domain_length
        )
        closure = basis.T @ (aligned_endpoint - state) / state_scale
        displacement = state - reference_state
        time_phase = float(
            np.dot(displacement, time_tangent) / time_scale
        )
        space_phase = float(
            np.dot(displacement, space_tangent_value) / space_scale
        )
        return np.concatenate([closure, [time_phase, space_phase]])

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        x_scale="jac",
        verbose=0,
    )
    state, period, spatial_shift_value = unpack(result.x)
    endpoint = endpoint_function(state, period, n_steps)
    aligned_endpoint = spectral_shift(
        endpoint, -spatial_shift_value, domain_length
    )
    full_closure = aligned_endpoint - state
    relative_defect = float(
        np.linalg.norm(full_closure) / (state_scale + 1e-15)
    )
    mean_defect = float(
        abs(np.mean(endpoint) - np.mean(state))
        / (np.std(state) + 1e-15)
    )
    displacement = state - reference_state
    time_phase = float(np.dot(displacement, time_tangent) / time_scale)
    space_phase = float(
        np.dot(displacement, space_tangent_value) / space_scale
    )
    state_correction = float(np.linalg.norm(displacement) / state_scale)
    speed_norm = float(np.linalg.norm(rhs_function(state)))
    converged = bool(
        result.success
        and relative_defect < defect_tolerance
        and speed_norm > 1e-6
        and state_correction < 1.0
    )
    return OrbitRefinement(
        state=state,
        period=period,
        spatial_shift=wrap_shift(spatial_shift_value, domain_length),
        endpoint=endpoint,
        optimizer_success=bool(result.success),
        converged=converged,
        nfev=int(result.nfev),
        cost=float(result.cost),
        relative_closure_defect=relative_defect,
        mean_closure_defect=mean_defect,
        time_phase_defect=time_phase,
        space_phase_defect=space_phase,
        relative_state_correction=state_correction,
        speed_norm=speed_norm,
        message=str(result.message),
    )


def orbit_record(prefix, refinement):
    """Flatten an OrbitRefinement for a CSV row."""
    return {
        f"{prefix}_optimizer_success": refinement.optimizer_success,
        f"{prefix}_converged": refinement.converged,
        f"{prefix}_period": refinement.period,
        f"{prefix}_spatial_shift": refinement.spatial_shift,
        f"{prefix}_closure_defect": refinement.relative_closure_defect,
        f"{prefix}_mean_defect": refinement.mean_closure_defect,
        f"{prefix}_time_phase_defect": refinement.time_phase_defect,
        f"{prefix}_space_phase_defect": refinement.space_phase_defect,
        f"{prefix}_state_correction": refinement.relative_state_correction,
        f"{prefix}_speed_norm": refinement.speed_norm,
        f"{prefix}_nfev": refinement.nfev,
        f"{prefix}_cost": refinement.cost,
        f"{prefix}_message": refinement.message,
    }
