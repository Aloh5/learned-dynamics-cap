"""Zero-Newton KS candidate from a regular-grid sensor surrogate.

The surrogate Y' = f(Y) is fitted to sensor time series at x_j = jL/(M+1)
exactly as in sensor_experiment.py. On this grid the DST Y = S a is a fixed
invertible linear map, so f is rewritten as g(a) = S^{-1} f(S a) before the
orbit search; a periodic orbit of g is the DST of a periodic orbit of f.
This lets the station surrogate use the same recurrence search, continuation
to 160 temporal harmonics, symmetry handling, and zero-Newton proof driver as
generate_surrogate_candidate.py. The true equation is used only to generate
the sensor data and, after freezing, for diagnostics.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import savemat

from ks_neumann import (NeumannKSConfig, generate_attractor_trajectory,
    scan_ordinary_recurrences, resample_segment_to_phase_grid,
    refine_ordinary_orbit, refine_ordinary_orbit_matrix_free,
    time_fourier_coefficients, time_values_from_fourier, collocation_residual,
    oversampled_residual, independent_flow_closure)
from surrogate_neumann import QuadraticSurrogate
from scan_block_proof import project_half_shift_symmetry
from sensor_experiment import (HERE, simulate_truth, sensor_sites, sine_matrix,
    fit_polynomial_vector_field, NodalQuadratic, cap_distance)

TEMPORAL_STAGES = [16, 24, 40, 64, 80, 96, 128, 160]


def finite_difference(values, h, order):
    """Equation (2) for order 2; fourth-order central differences otherwise,
    with the second-order one-sided formulas at the two ends."""
    d = np.empty_like(values)
    d[1:-1] = (values[2:] - values[:-2])/(2*h)
    d[0] = (-3*values[0] + 4*values[1] - values[2])/(2*h)
    d[-1] = (3*values[-1] - 4*values[-2] + values[-3])/(2*h)
    if order == 4:
        d[2:-2] = (-values[4:] + 8*values[3:-1] - 8*values[1:-3] + values[:-4])/(12*h)
    return d


def to_sine_coordinates(nodal, count):
    """g(a) = S^{-1} f(S a) with S_jk = sin(jk pi/(M+1)), S^{-1} = 2/(M+1) S."""
    S = np.sin(np.outer(np.arange(1, count + 1), np.arange(1, count + 1))*np.pi/(count + 1))
    Sinv = (2/(count + 1))*S
    constant = Sinv @ nodal.c
    linear = Sinv @ nodal.A @ S
    quadratic = np.einsum('op,prs,ri,sj->oij', Sinv, nodal.B, S, S, optimize=True)
    # Reflection equivariance, (Ra)_k = (-1)^k a_k, as in fit_surrogate.
    parity = (-1.)**np.arange(1, count + 1)
    constant *= (1 + parity)/2
    linear *= (1 + parity[:, None]*parity[None, :])/2
    quadratic *= (1 + parity[:, None, None]*parity[None, :, None]*parity[None, None, :])/2
    return QuadraticSurrogate(constant, linear, quadratic)


def pad(coefficients, temporal, spatial):
    out = np.zeros((2*temporal + 1, spatial), complex)
    m = (len(coefficients) - 1)//2
    k = min(spatial, coefficients.shape[1])
    out[temporal - m:temporal + m + 1, :k] = coefficients[:, :k]
    return out


def continue_orbit(seed, period, model, count, log):
    coefficients = None
    for m in TEMPORAL_STAGES:
        config = NeumannKSConfig(spatial_modes=count, temporal_modes=m,
                                 full_grid_nodes=max(128, 4*count), surrogate=model)
        if coefficients is not None:
            seed = time_values_from_fourier(pad(coefficients, m, count))
        if m <= 40:
            orbit = refine_ordinary_orbit(seed, period, config, max_nfev=260)
        else:
            orbit = refine_ordinary_orbit_matrix_free(seed, 2*np.pi/period, config,
                                                      max_nfev=60, tight=(m == 160))
        row = dict(temporal_modes=m, spatial_modes=count, period=orbit.period,
                   surrogate_residual_rms=orbit.residual_l2,
                   surrogate_residual_max=orbit.residual_linf, nfev=orbit.nfev)
        log.append(row); print('STAGE', json.dumps(row), flush=True)
        if not orbit.optimizer_success or orbit.residual_linf > 1e-7:
            raise RuntimeError(f'surrogate continuation failed at {m}')
        period = orbit.period
        coefficients = time_fourier_coefficients(orbit.time_values)
    return coefficients, period, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensors', type=int, required=True)
    ap.add_argument('--fd-order', type=int, choices=(2, 4), default=4)
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--tries', type=int, default=4)
    args = ap.parse_args()
    count = args.sensors
    out = args.output_dir or HERE/'outputs_sensor_spectral'/f'M{count}_fd{args.fd_order}'
    out.mkdir(parents=True, exist_ok=True)

    dt = 0.1
    data, _ = simulate_truth(48, dt, 300., 3000.)
    x = sensor_sites('regular', count)
    Y = data @ sine_matrix(x, 48).T
    Ydot = finite_difference(Y, dt, args.fd_order)
    split = int(0.7*len(Y))
    fits = []
    for lam in (0., 1e-10, 1e-8, 1e-6):
        model = fit_polynomial_vector_field(Y[:split], Ydot[:split], degree=2,
                                            regularization=lam, normalize=True)
        err = np.linalg.norm(model.predict(Y[split:]) - Ydot[split:])/np.linalg.norm(Ydot[split:])
        fits.append((err, lam, model))
    chosen = None
    search = None
    for err, lam, model in sorted(fits, key=lambda t: t[0]):
        g = to_sine_coordinates(NodalQuadratic(model), count)
        config = NeumannKSConfig(spatial_modes=count, temporal_modes=16, sample_time=800.,
                                 full_grid_nodes=max(128, 4*count), surrogate=g)
        try:
            _, trajectory = generate_attractor_trajectory(config)
            bounded = bool(np.max(np.abs(trajectory)) < 10)
        except RuntimeError:
            bounded = False
        print('FIT', lam, err, 'bounded' if bounded else 'diverged', flush=True)
        if bounded:
            chosen, search = (err, lam, g), trajectory
            break
    if chosen is None:
        raise SystemExit('every surrogate rollout diverged')
    err, lam, g = chosen
    g.save(out/f'surrogate_M{count}.npz')

    returns = scan_ordinary_recurrences(search, 0.1, min_period=20, max_period=120,
                                        max_candidates=30)
    returns.to_csv(out/'surrogate_recurrences.csv', index=False)
    stages = []
    for rank, row in returns.head(args.tries).iterrows():
        seed = resample_segment_to_phase_grid(
            search[int(row.start_index):int(row.end_index) + 1], 16)
        stages = []
        try:
            coefficients, period, config = continue_orbit(seed, float(row.period), g, count, stages)
            break
        except (RuntimeError, np.linalg.LinAlgError) as error:
            print('SEED FAILED', rank, row.period, error, flush=True)
    else:
        raise SystemExit('no recurrence continued to 160 harmonics')

    # Export exactly as generate_surrogate_candidate.py: pad to 60 spatial
    # modes (zeros above M), canonicalize symmetry/reality, then freeze.
    padded = pad(coefficients, 160, 60)
    projected = project_half_shift_symmetry(padded)
    projection_change = float(np.max(np.abs(projected - padded)))
    surrogate_residual = collocation_residual(time_values_from_fourier(coefficients),
                                              2*np.pi/period, config)
    values = time_values_from_fourier(projected)
    data_out = dict(frequency=2*np.pi/period, period=period, velocity_time_fourier=projected,
        velocity_time_values=values, reference_time_values=values, interval_length=22.,
        extension_length=44., spatial_modes=60, temporal_modes=160, newton_corrections=0,
        true_system_optimizer_calls=0, candidate_origin='data_fitted_surrogate_only',
        symmetry_projection_max_change=projection_change, sensors=count,
        fd_order=args.fd_order, ridge=lam)
    savemat(out/'ks_neumann_cap_candidate.mat', data_out)

    # Diagnostics only after freezing.
    true_config = NeumannKSConfig(spatial_modes=60, temporal_modes=160, full_grid_nodes=240)
    _, residual = oversampled_residual(values, 2*np.pi/period, true_config)
    closure = independent_flow_closure(values, period, true_config)
    refined = refine_ordinary_orbit_matrix_free(values, 2*np.pi/period, true_config,
                                                max_nfev=60, tight=True)
    refined_coefficients = project_half_shift_symmetry(time_fourier_coefficients(refined.time_values))
    record = dict(sensors=count, fd_order=args.fd_order, ridge=lam,
        heldout_derivative_error=err, surrogate_period=period,
        surrogate_retained_residual_rms=float(np.sqrt(np.mean(surrogate_residual**2))),
        symmetry_projection_max_change=projection_change,
        true_oversampled_residual_rms=float(np.sqrt(np.mean(residual**2))),
        true_oversampled_residual_max=float(np.max(np.abs(residual))),
        true_newton_residual_max=refined.residual_linf, true_refined_period=refined.period,
        cap_norm_distance_to_true_orbit=float(cap_distance(projected, period,
                                                          refined_coefficients, refined.period)),
        continuation=stages, true_system_optimizer_calls=0, newton_corrections=0, **closure)
    (out/'candidate_diagnostics.json').write_text(json.dumps(record, indent=2) + '\n')
    print('DONE', json.dumps({k: v for k, v in record.items() if k != 'continuation'}), flush=True)


if __name__ == '__main__':
    main()
