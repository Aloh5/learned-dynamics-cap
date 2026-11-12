"""Newton-refine a sensor-surrogate orbit in the true equation for CAP.

Rebuilds the regular-grid sensor surrogate of sensor_experiment.py, freezes its
DST-reconstructed candidates, keeps those with the half-period reflection
symmetry required by the KS proof, and continues the chosen one with Newton in
the true Galerkin equation up to the proof resolution (160 temporal, 60
spatial modes). The frozen sensor candidate is saved alongside, so the final
certificate can be compared with it. Floating-point only; the proof is run
separately by run_sensor_refined_proof.sh.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from ks_neumann import (NeumannKSConfig, refine_ordinary_orbit,
    refine_ordinary_orbit_matrix_free, time_fourier_coefficients,
    time_values_from_fourier, oversampled_residual, independent_flow_closure)
from scan_block_proof import project_half_shift_symmetry
from sensor_experiment import (HERE, simulate_truth, sensor_sites, sine_matrix,
    finite_difference, fit_polynomial_vector_field, NodalQuadratic, rk4,
    surrogate_orbits, reconstruct_dst, cap_distance)

STAGES = [(32, 32), (48, 36), (64, 40), (96, 44), (128, 52), (160, 60)]


def symmetry_defect(coefficients):
    projected = project_half_shift_symmetry(coefficients)
    return float(np.linalg.norm(projected - coefficients)/np.linalg.norm(coefficients))


def pad(coefficients, temporal, spatial):
    out = np.zeros((2*temporal + 1, spatial), complex)
    m = (len(coefficients) - 1)//2
    k = min(spatial, coefficients.shape[1])
    out[temporal - m:temporal + m + 1, :k] = coefficients[:, :k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensors', type=int, default=24)
    ap.add_argument('--output-dir', type=Path, default=HERE/'outputs_sensor_proof')
    ap.add_argument('--max-symmetry-defect', type=float, default=1e-2)
    args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    # Same settings as the sweep in sensor_experiment.py (defaults).
    dt = 0.1
    data, truth = simulate_truth(48, dt, 300., 3000.)
    x = sensor_sites('regular', args.sensors)
    Y = data @ sine_matrix(x, 48).T
    Ydot = finite_difference(Y, dt)
    split = int(0.7*len(Y))
    # Select by held-out derivative error among models whose rollout stays
    # bounded (manuscript section 2.2); error alone picked divergent models.
    fits = []
    for lam in (0., 1e-10, 1e-8, 1e-6):
        model = fit_polynomial_vector_field(Y[:split], Ydot[:split], degree=2,
                                            regularization=lam, normalize=True)
        err = np.linalg.norm(model.predict(Y[split:]) - Ydot[split:])/np.linalg.norm(Ydot[split:])
        fits.append((err, lam, model))
    for err, lam, model in sorted(fits, key=lambda t: t[0]):
        f = NodalQuadratic(model)
        traj, bounded = rk4(f, Y[split], 0.005, int(round(2300/0.005)), int(round(dt/0.005)))
        print('FIT', lam, err, 'bounded' if bounded else 'diverged', flush=True)
        if bounded:
            best = (err, lam, model)
            break
    else:
        raise RuntimeError('every surrogate rollout diverged')
    orbits = surrogate_orbits(f, traj[3000:], dt, [16, 32], tries=6)

    rows = []
    for index, orbit in enumerate(orbits):
        a = reconstruct_dst(time_fourier_coefficients(orbit['values']), args.sensors)
        rows.append(dict(orbit=index, surrogate_period=orbit['period'],
                         symmetry_defect=symmetry_defect(a), candidate=a))
        print('ORBIT', index, orbit['period'], rows[-1]['symmetry_defect'], flush=True)
    (out/'orbit_symmetry.json').write_text(json.dumps(
        [{k: v for k, v in r.items() if k != 'candidate'} for r in rows], indent=2) + '\n')
    symmetric = [r for r in rows if r['symmetry_defect'] < args.max_symmetry_defect]
    if not symmetric:
        raise SystemExit('No sensor orbit has the half-period reflection symmetry; '
                         'the existing KS proof cannot be applied.')
    chosen = min(symmetric, key=lambda r: r['symmetry_defect'])
    frozen = chosen['candidate']
    period = chosen['surrogate_period']
    np.savez(out/'sensor_frozen_candidate.npz', velocity_time_fourier=frozen,
             period=period, sensors=args.sensors, sites=x, ridge=best[1],
             symmetry_defect=chosen['symmetry_defect'])

    # Newton continuation in the TRUE equation (surrogate=None).
    coefficients = project_half_shift_symmetry(pad(frozen, 32, 32))
    history = []
    for temporal, spatial in STAGES:
        config = NeumannKSConfig(spatial_modes=spatial, temporal_modes=temporal,
                                 full_grid_nodes=max(128, 4*spatial))
        seed = time_values_from_fourier(pad(coefficients, temporal, spatial))
        if temporal <= 64:
            orbit = refine_ordinary_orbit(seed, period, config, max_nfev=200)
        else:
            orbit = refine_ordinary_orbit_matrix_free(seed, 2*np.pi/period, config,
                                                      max_nfev=60, tight=(temporal == 160))
        coefficients = project_half_shift_symmetry(time_fourier_coefficients(orbit.time_values))
        period = orbit.period
        row = dict(temporal_modes=temporal, spatial_modes=spatial, period=period,
                   residual_rms=orbit.residual_l2, residual_max=orbit.residual_linf,
                   nfev=orbit.nfev, success=orbit.optimizer_success)
        history.append(row); print('STAGE', json.dumps(row), flush=True)
        if orbit.residual_linf > 1e-7:
            raise RuntimeError(f'True-equation refinement failed at {temporal},{spatial}')

    values = time_values_from_fourier(coefficients)
    _, residual = oversampled_residual(values, 2*np.pi/period, config)
    closure = independent_flow_closure(values, period, config)
    distance = cap_distance(frozen, chosen['surrogate_period'], coefficients, period)
    data = dict(frequency=2*np.pi/period, period=period, velocity_time_fourier=coefficients,
                velocity_time_values=values, reference_time_values=values,
                interval_length=22., extension_length=44., spatial_modes=60,
                temporal_modes=160, newton_corrections=len(STAGES),
                true_system_optimizer_calls=len(STAGES),
                candidate_origin='sensor_surrogate_then_true_newton')
    savemat(out/'ks_neumann_cap_candidate.mat', data)
    record = dict(sensors=args.sensors, ridge=best[1], frozen_surrogate_period=chosen['surrogate_period'],
                  frozen_symmetry_defect=chosen['symmetry_defect'], refined_period=period,
                  true_oversampled_residual_rms=float(np.sqrt(np.mean(residual**2))),
                  true_oversampled_residual_max=float(np.max(np.abs(residual))),
                  cap_norm_distance_frozen_to_refined=float(distance),
                  continuation=history, **closure)
    (out/'candidate_diagnostics.json').write_text(json.dumps(record, indent=2) + '\n')
    print('DONE', json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
