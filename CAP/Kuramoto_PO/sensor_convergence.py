"""Sensor-count study for the Neumann KS station surrogate.

For one sensor count M (regular sites x_j = jL/(M+1)) this script

1. fits the complete quadratic Y' = f(Y) to the sensor time series, choosing
   the ridge parameter by held-out derivative error among models whose RK4
   rollout stays bounded;
2. finds periodic orbits of f by harmonic balance (16 then 32 harmonics),
   seeded by close returns of the rollout;
3. reconstructs and freezes each candidate by the discrete sine transform;
4. measures the candidate in the true Galerkin collocation equation
   (32 harmonics, 48 sine modes): the initial defect, then plain Newton
   iterations, recording every residual and every correction in norm (25),
   until the residual reaches 1e-10 or Newton fails.

The truth trajectory is cached so that every M uses identical data. Floating
point only; nothing here is a proof.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ks_neumann import (NeumannKSConfig, collocation_residual, nonlinear_sine_jacobian,
    spatial_wavenumbers, temporal_differentiation_matrix, time_fourier_coefficients,
    time_values_from_fourier, oversampled_residual)
from sensor_experiment import (HERE, CERTIFIED_PERIOD, Q_SCALE, simulate_truth, sensor_sites,
    sine_matrix, NodalQuadratic, rk4, surrogate_orbits, reconstruct_dst, cap_norm,
    fit_polynomial_vector_field)
from sensor_spectral_candidate import finite_difference

OUT = HERE/'outputs_sensor_convergence'
TRUTH_MODES, DT, BURN, DURATION = 48, 0.1, 300., 3000.
NEWTON_TEMPORAL, NEWTON_SPATIAL = 32, 48


def truth_data():
    path = OUT/'truth.npz'
    if not path.exists():
        data, _ = simulate_truth(TRUTH_MODES, DT, BURN, DURATION)
        np.savez(path, data=data)
    return np.load(path)['data']


def pad(coefficients, temporal, spatial):
    out = np.zeros((2*temporal + 1, spatial), complex)
    m = (len(coefficients) - 1)//2
    k = min(spatial, coefficients.shape[1])
    out[temporal - m:temporal + m + 1, :k] = coefficients[:, :k]
    return out


def newton_true(candidate, period, tol=1e-10, max_iter=30):
    """Plain Newton for the phase-augmented true collocation equation.

    Returns the iteration history: max residual before each step and the size
    of each correction in norm (25) (state and frequency parts together).
    """
    config = NeumannKSConfig(spatial_modes=NEWTON_SPATIAL, full_grid_nodes=4*NEWTON_SPATIAL)
    values = time_values_from_fourier(pad(candidate, NEWTON_TEMPORAL, NEWTON_SPATIAL))
    frequency = 2*np.pi/period
    n, m = values.shape
    D = temporal_differentiation_matrix(n)
    reference, tangent = values.copy(), D @ values
    diagonal = spatial_wavenumbers(config)**4 - spatial_wavenumbers(config)**2
    history = []
    for iteration in range(max_iter + 1):
        F = collocation_residual(values, frequency, config)
        residual = float(np.max(np.abs(F)))
        history.append(dict(iteration=iteration, residual_max=residual))
        if residual < tol or residual > 1e4 or not np.isfinite(residual) or iteration == max_iter:
            break
        J = np.zeros((n*m + 1, n*m + 1))
        J[:n*m, :n*m] = frequency*np.kron(D, np.eye(m))
        for t, state in enumerate(values):
            J[t*m:(t + 1)*m, t*m:(t + 1)*m] += np.diag(diagonal) + nonlinear_sine_jacobian(state, config)
        J[:n*m, -1] = (D @ values).ravel()
        J[-1, :n*m] = tangent.ravel()
        rhs = -np.r_[F.ravel(), np.sum((values - reference)*tangent)]
        step = np.linalg.solve(J, rhs)
        dv = step[:-1].reshape(n, m)
        history[-1]['correction_cap_norm'] = float(cap_norm(step[-1]/Q_SCALE**2,
                                                            time_fourier_coefficients(dv)))
        values, frequency = values + dv, frequency + step[-1]
    converged = history[-1]['residual_max'] < tol
    return history, converged, time_fourier_coefficients(values), 2*np.pi/frequency


def run(M, fd_order, out):
    data = truth_data()
    x = sensor_sites('regular', M)
    Y = data @ sine_matrix(x, TRUTH_MODES).T
    Ydot = finite_difference(Y, DT, fd_order)
    split = int(0.7*len(Y))
    surrogate_dt = 0.005 if M <= 32 else 0.002
    fits = []
    for lam in (0., 1e-10, 1e-8, 1e-6):
        model = fit_polynomial_vector_field(Y[:split], Ydot[:split], degree=2,
                                            regularization=lam, normalize=True)
        err = np.linalg.norm(model.predict(Y[split:]) - Ydot[split:])/np.linalg.norm(Ydot[split:])
        fits.append((float(err), lam, model))
    record = dict(sensors=M, fd_order=fd_order, surrogate_dt=surrogate_dt,
                  heldout_error_by_ridge={str(lam): err for err, lam, _ in fits},
                  best_heldout_error=min(f[0] for f in fits), ridge_trials=[])
    chosen = None
    for err, lam, model in sorted(fits, key=lambda t: t[0]):
        f = NodalQuadratic(model)
        traj, bounded = rk4(f, Y[split], surrogate_dt, int(round(2300/surrogate_dt)),
                            int(round(DT/surrogate_dt)))
        record['ridge_trials'].append(dict(ridge=lam, heldout_error=err, bounded=bool(bounded),
                                           rollout_time=float((len(traj) - 1)*DT)))
        print('FIT', M, lam, err, bounded, flush=True)
        if bounded:
            chosen = (err, lam, f, traj)
            break
    if chosen is None:
        record['status'] = 'every rollout diverged'
        return record
    err, lam, f, traj = chosen
    record.update(ridge=lam, heldout_error=err,
                  rollout_std_ratio=float(np.std(traj[3000:])/np.std(Y)))
    orbits = surrogate_orbits(f, traj[3000:], DT, [16, 32], tries=8)
    record['orbits'] = []
    arrays = {}
    for index, orbit in enumerate(orbits):
        candidate = reconstruct_dst(time_fourier_coefficients(orbit['values']), M)
        config = NeumannKSConfig(spatial_modes=NEWTON_SPATIAL, full_grid_nodes=4*NEWTON_SPATIAL)
        _, over = oversampled_residual(time_values_from_fourier(
            pad(candidate, NEWTON_TEMPORAL, NEWTON_SPATIAL)), 2*np.pi/orbit['period'], config)
        history, converged, refined, period = newton_true(candidate, orbit['period'])
        reference = pad(refined, NEWTON_TEMPORAL, NEWTON_SPATIAL)
        frozen = pad(candidate, NEWTON_TEMPORAL, NEWTON_SPATIAL)
        distance = cap_norm((2*np.pi/orbit['period'] - 2*np.pi/period)/Q_SCALE**2, frozen - reference)
        row = dict(orbit=index, surrogate_period=orbit['period'], harmonic_balance_rms=orbit['rms'],
                   initial_defect_max=history[0]['residual_max'],
                   initial_defect_oversampled_rms=float(np.sqrt(np.mean(over**2))),
                   first_newton_correction=history[0].get('correction_cap_norm'),
                   newton_converged=bool(converged),
                   newton_corrections=len(history) - 1 if converged else None,
                   true_period=period if converged else None,
                   distance_to_true_orbit=float(distance) if converged else None,
                   newton_history=history)
        record['orbits'].append(row)
        arrays[f'candidate_{index}'] = candidate
        arrays[f'refined_{index}'] = refined
        print('ORBIT', M, json.dumps({k: v for k, v in row.items() if k != 'newton_history'}), flush=True)
    np.savez(out/f'M{M}_candidates.npz', **arrays)
    record['status'] = 'ok' if orbits else 'no periodic orbit found'
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensors', type=int, required=True)
    ap.add_argument('--fd-order', type=int, choices=(2, 4), default=2)
    args = ap.parse_args()
    out = OUT/f'fd{args.fd_order}'
    out.mkdir(parents=True, exist_ok=True)
    record = run(args.sensors, args.fd_order, out)
    (out/f'M{args.sensors}.json').write_text(json.dumps(record, indent=2) + '\n')
    print('DONE', args.sensors, record['status'], flush=True)


if __name__ == '__main__':
    main()
