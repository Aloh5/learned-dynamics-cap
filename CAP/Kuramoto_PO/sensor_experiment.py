"""Periodic Neumann KS candidates from synchronized sensor time series.

Implements the "weather station" experiment of the 2 Nov 2026 note:

1. simulate the true Neumann KS velocity equation and record u at M fixed sites;
2. estimate time derivatives by finite differences and fit a complete quadratic
   vector field Y' = f(Y) in nodal coordinates (no Galerkin field is used);
3. find a periodic orbit of f by harmonic balance (collocation), seeded by a
   close return of a surrogate trajectory;
4. turn the temporal Fourier coefficients b_n in R^M into spatial sine
   coefficients, either by the discrete sine transform (regular grid) or by
   natural cubic splines followed by sine projection (any grid);
5. freeze that space-time candidate and only then evaluate it in the true
   PDE: residual, Newton refinement in the true Galerkin equation, and the
   distance in the CAP norm (25) to the true orbit it converges to.

Floating-point diagnostics only; nothing here is a proof.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.optimize import least_squares

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dpsr.polynomial import fit_polynomial_vector_field
from ks_neumann import (NeumannKSConfig, default_initial_coefficients,
    simulate_velocity_sine, scan_ordinary_recurrences,
    resample_segment_to_phase_grid, temporal_differentiation_matrix,
    time_fourier_coefficients, time_values_from_fourier, oversampled_residual,
    refine_ordinary_orbit)

HERE = Path(__file__).resolve().parent
L = 22.0
Q_SCALE = np.pi / L
CERTIFIED_PERIOD = 72.70550498125212


# ---------------------------------------------------------------- truth data
def simulate_truth(modes, dt, burn, duration, seed=7):
    config = NeumannKSConfig(spatial_modes=modes, full_grid_nodes=max(128, 4*modes),
                             output_dt=dt, random_seed=seed)
    initial = np.zeros(modes)
    initial[:20] = default_initial_coefficients(replace(config, spatial_modes=20))
    times, values = simulate_velocity_sine(initial, burn + duration, config)
    start = int(round(burn/dt))
    return values[start:], config


def sensor_sites(layout, count, seed=0):
    if layout == 'regular':
        return L*np.arange(1, count + 1)/(count + 1)
    rng = np.random.default_rng(seed)
    gap = L/(4*(count + 1))
    while True:
        x = np.sort(rng.uniform(gap, L - gap, count))
        if np.min(np.diff(x)) > gap:
            return x


def sine_matrix(x, modes):
    return np.sin(np.outer(x, np.arange(1, modes + 1))*np.pi/L)


def finite_difference(values, h):
    """Second-order differences, equation (2) of the manuscript."""
    d = np.empty_like(values)
    d[1:-1] = (values[2:] - values[:-2])/(2*h)
    d[0] = (-3*values[0] + 4*values[1] - values[2])/(2*h)
    d[-1] = (3*values[-1] - 4*values[-2] + values[-3])/(2*h)
    return d


# ------------------------------------------------------------ surrogate model
class NodalQuadratic:
    """Quadratic f(Y) = c + A Y + B(Y,Y) unpacked from a fitted PolynomialModel."""
    def __init__(self, model):
        coefficients = model.coefficients()
        powers = model.feature_library.powers_
        m = powers.shape[1]
        degree = powers.sum(axis=1)
        self.c = coefficients[:, degree == 0][:, 0]
        self.A = np.zeros((m, m))
        self.B = np.zeros((m, m, m))
        for column, p in zip(coefficients.T, powers):
            idx = np.flatnonzero(p)
            if p.sum() == 1:
                self.A[:, idx[0]] = column
            elif p.sum() == 2:
                i, j = (idx[0], idx[0]) if len(idx) == 1 else idx
                if i == j:
                    self.B[:, i, i] = column
                else:
                    self.B[:, i, j] = self.B[:, j, i] = column/2

    def __call__(self, y):
        return self.c + y @ self.A.T + np.einsum('oij,...i,...j->...o', self.B, y, y, optimize=True)

    def jacobian(self, y):
        return self.A + 2*np.einsum('oij,...j->...oi', self.B, y, optimize=True)


def rk4(f, y0, dt, steps, save_every):
    y = np.array(y0, float)
    out = [y.copy()]
    for step in range(1, steps + 1):
        k1 = f(y); k2 = f(y + dt/2*k1); k3 = f(y + dt/2*k2); k4 = f(y + dt*k3)
        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e3:
            return np.array(out), False
        if step % save_every == 0:
            out.append(y.copy())
    return np.array(out), True


# ----------------------------------------------------------- harmonic balance
def harmonic_balance(f, seed_values, period, max_nfev=200):
    """Collocation for Y' = f(Y) with unknown frequency and one phase condition."""
    reference = np.asarray(seed_values, float)
    n, m = reference.shape
    D = temporal_differentiation_matrix(n)
    tangent = D @ reference
    tscale = float(np.sum(tangent**2))

    def unpack(p):
        return p[:-1].reshape(n, m), float(np.exp(p[-1]))

    def residual(p):
        y, w = unpack(p)
        return np.r_[(w*(D @ y) - f(y)).ravel(), np.sum((y - reference)*tangent)/tscale]

    def jacobian(p):
        y, w = unpack(p)
        total = n*m
        J = np.zeros((total + 1, total + 1))
        J[:total, :total] = w*np.kron(D, np.eye(m))
        blocks = f.jacobian(y)
        for t in range(n):
            J[t*m:(t + 1)*m, t*m:(t + 1)*m] -= blocks[t]
        J[:total, -1] = (w*(D @ y)).ravel()
        J[-1, :total] = tangent.ravel()/tscale
        return J

    result = least_squares(residual, np.r_[reference.ravel(), np.log(2*np.pi/period)],
                           jac=jacobian, xtol=1e-14, ftol=1e-14, gtol=1e-14,
                           max_nfev=max_nfev, x_scale='jac')
    y, w = unpack(result.x)
    r = residual(result.x)[:-1]
    return y, 2*np.pi/w, float(np.sqrt(np.mean(r**2)))


def pad_time(coefficients, modes):
    old = (len(coefficients) - 1)//2
    out = np.zeros((2*modes + 1, coefficients.shape[1]), complex)
    out[modes - old:modes + old + 1] = coefficients
    return out


def surrogate_orbits(f, trajectory, dt, harmonics, tries=8):
    returns = scan_ordinary_recurrences(trajectory, dt, min_period=20, max_period=120,
                                        max_candidates=30)
    found = []
    for _, row in returns.head(tries).iterrows():
        seed = resample_segment_to_phase_grid(
            trajectory[int(row.start_index):int(row.end_index) + 1], harmonics[0])
        period = float(row.period)
        try:
            for h in harmonics:
                if len(seed) != 2*h + 1:
                    seed = time_values_from_fourier(pad_time(time_fourier_coefficients(seed), h))
                seed, period, rms = harmonic_balance(f, seed, period)
        except (np.linalg.LinAlgError, ValueError):
            continue
        amplitude = np.ptp(seed, axis=0).max()
        if rms < 1e-9 and amplitude > 1e-2 and 20 < period < 150:
            if not any(abs(period - o['period']) < 1e-3*period for o in found):
                found.append(dict(values=seed, period=period, rms=rms,
                                  seed_period=float(row.period)))
    return found


# ----------------------------------------------------------- reconstructions
def reconstruct_dst(b, count):
    """Equation (6): exact sine interpolant on x_j = jL/(M+1)."""
    S = np.sin(np.outer(np.arange(1, count + 1), np.arange(1, count + 1))*np.pi/(count + 1))
    return (2/(count + 1))*(b @ S)


def reconstruct_collocation(b, x):
    """Nonuniform sine collocation: solve sum_m a_m sin(m pi x_j/L) = b_j."""
    S = sine_matrix(x, len(x))
    return np.linalg.solve(S, b.T).T, float(np.linalg.cond(S))


def reconstruct_spline(b, x, modes, gauss=24):
    """Equation (4): natural cubic spline through (0,0),(x_k,b_k),(L,0), then sine projection."""
    knots = np.r_[0.0, x, L]
    values = np.hstack([np.zeros((len(b), 1)), b, np.zeros((len(b), 1))])
    spline = CubicSpline(knots, values.T, bc_type='natural', axis=0)
    nodes, weights = np.polynomial.legendre.leggauss(gauss)
    left, right = knots[:-1], knots[1:]
    xs = ((right - left)[:, None]*(nodes[None, :] + 1)/2 + left[:, None]).ravel()
    ws = ((right - left)[:, None]*weights[None, :]/2).ravel()
    basis = np.sin(np.outer(xs, np.arange(1, modes + 1))*np.pi/L)
    return (2/L)*((spline(xs)*ws[:, None]).T @ basis)


# --------------------------------------------------------- true-PDE testing
def cap_norm(f, coefficients):
    """Norm (25) in scaled variables v = u/q, f = omega/q^2; coefficients complex-centered."""
    m = (len(coefficients) - 1)//2
    v = coefficients/Q_SCALE
    positive = v[m + 1:]
    return abs(f) + np.sum(np.abs(v[m].real)) + np.sum(np.abs(2*positive.real) + np.abs(2*positive.imag))


def cap_distance(a, period_a, b, period_b):
    temporal = max(len(a), len(b))//2
    spatial = max(a.shape[1], b.shape[1])
    pa = np.zeros((2*temporal + 1, spatial), complex); pb = pa.copy()
    for src, dst in ((a, pa), (b, pb)):
        m = (len(src) - 1)//2
        dst[temporal - m:temporal + m + 1, :src.shape[1]] = src
    df = (2*np.pi/period_a - 2*np.pi/period_b)/Q_SCALE**2
    return cap_norm(df, pa - pb)


def align_phase(reference, period_ref, candidate, period_c):
    """Best temporal shift of candidate onto reference (for comparing orbits)."""
    temporal = (len(candidate) - 1)//2
    n = np.arange(-temporal, temporal + 1)[:, None]
    grid = np.linspace(0, 2*np.pi, 2048, endpoint=False)
    m = (len(reference) - 1)//2
    best = None
    for shift in grid:
        shifted = candidate*np.exp(1j*n*shift)
        d = cap_distance(reference, period_ref, shifted, period_c)
        if best is None or d < best[0]:
            best = (d, shift)
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda s: cap_distance(reference, period_ref, candidate*np.exp(1j*n*s), period_c),
                          bracket=(best[1] - 2*np.pi/2048, best[1], best[1] + 2*np.pi/2048))
    return candidate*np.exp(1j*n*res.x), float(res.fun)


def evaluate_candidate(coefficients, period, true_modes):
    """Freeze candidate, then residual + true Newton refinement in a K-mode Galerkin equation."""
    temporal = (len(coefficients) - 1)//2
    padded = np.zeros((len(coefficients), true_modes), complex)
    k = min(true_modes, coefficients.shape[1])
    padded[:, :k] = coefficients[:, :k]
    config = NeumannKSConfig(spatial_modes=true_modes, full_grid_nodes=max(128, 4*true_modes))
    values = time_values_from_fourier(padded)
    _, residual = oversampled_residual(values, 2*np.pi/period, config)
    orbit = refine_ordinary_orbit(values, period, config, max_nfev=60)
    refined = time_fourier_coefficients(orbit.time_values)
    return dict(
        candidate_period=period,
        true_residual_rms=float(np.sqrt(np.mean(residual**2))),
        true_residual_max=float(np.max(np.abs(residual))),
        newton_converged=bool(orbit.residual_linf < 1e-9),
        newton_residual_max=orbit.residual_linf,
        refined_period=orbit.period,
        cap_distance_to_true_orbit=float(cap_distance(padded, period, refined, orbit.period)),
        relative_l2_distance=float(np.linalg.norm(padded - refined)/np.linalg.norm(refined)),
    ), refined, orbit.period


def certified_reference(temporal, modes):
    source = loadmat(HERE/'outputs_surrogate'/'ks_neumann_cap_candidate.mat')
    c = source['velocity_time_fourier']
    m = (len(c) - 1)//2
    return c[m - temporal:m + temporal + 1, :modes], float(source['period'].item()), c


# ------------------------------------------------------------------ driver
def run_case(layout, count, data, truth_config, args, certified_full):
    x = sensor_sites(layout, count, seed=args.site_seed)
    Y = data @ sine_matrix(x, truth_config.spatial_modes).T
    Ydot = finite_difference(Y, args.dt)
    split = int(0.7*len(Y))
    rows, models = [], {}
    for lam in args.ridge:
        model = fit_polynomial_vector_field(Y[:split], Ydot[:split], degree=2,
                                            regularization=lam, normalize=True)
        err = model.predict(Y[split:]) - Ydot[split:]
        rows.append((float(np.linalg.norm(err)/np.linalg.norm(Ydot[split:])), lam))
        models[lam] = model
    val_err, lam = min(rows)
    f = NodalQuadratic(models[lam])
    traj, bounded = rk4(f, Y[split], args.surrogate_dt, int(round(args.rollout/args.surrogate_dt)),
                        int(round(args.dt/args.surrogate_dt)))
    record = dict(layout=layout, sensors=count, sites=x.round(4).tolist(), ridge=lam,
                  validation_relative_derivative_error=val_err, rollout_bounded=bounded,
                  rollout_length=float((len(traj) - 1)*args.dt))
    burn = int(round(args.rollout_burn/args.dt))
    if not bounded or len(traj) <= burn + 1500:
        record['status'] = 'surrogate rollout not usable'
        return record, []
    record['rollout_std_ratio'] = float(np.std(traj[burn:])/np.std(Y))
    orbits = surrogate_orbits(f, traj[burn:], args.dt, args.harmonics, tries=args.tries)
    record['surrogate_orbits_found'] = len(orbits)
    results = []
    for index, orbit in enumerate(orbits):
        b = time_fourier_coefficients(orbit['values'])
        methods = {}
        if layout == 'regular':
            methods['dst'] = reconstruct_dst(b, count)
        else:
            a, cond = reconstruct_collocation(b, x)
            methods['sine_collocation'] = a
            record['collocation_condition_number'] = cond
        for q in args.spline_modes:
            methods[f'spline_Q{q}'] = reconstruct_spline(b, x, q)
        for name, a in methods.items():
            stats, refined, refined_period = evaluate_candidate(a, orbit['period'], args.true_modes)
            stats.update(layout=layout, sensors=count, orbit=index, method=name,
                         surrogate_period=orbit['period'], surrogate_hb_rms=orbit['rms'],
                         seed_period=orbit['seed_period'])
            same = abs(refined_period - CERTIFIED_PERIOD) < 1e-3 and stats['newton_converged']
            stats['refines_to_certified_orbit'] = bool(same)
            if same:
                ref = certified_full[(len(certified_full) - 1)//2 - (len(refined) - 1)//2:
                                     (len(certified_full) - 1)//2 + (len(refined) - 1)//2 + 1,
                                     :refined.shape[1]]
                _, d = align_phase(ref, CERTIFIED_PERIOD, refined, refined_period)
                stats['refined_vs_certified_cap_distance'] = d
            results.append(stats)
            print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, list)}), flush=True)
    return record, results


def reconstruction_only(certified, period, layouts, counts, spline_modes, true_modes, site_seed):
    """Control: sample the certified orbit itself at the sensors and reconstruct."""
    rows = []
    temporal = (len(certified) - 1)//2
    reference = np.zeros((len(certified), true_modes), complex)
    reference[:, :min(true_modes, certified.shape[1])] = certified[:, :true_modes]
    for layout in layouts:
        for count in counts:
            x = sensor_sites(layout, count, seed=site_seed)
            b = certified @ sine_matrix(x, certified.shape[1]).T
            methods = {f'spline_Q{q}': reconstruct_spline(b, x, q) for q in spline_modes}
            if layout == 'regular':
                methods['dst'] = reconstruct_dst(b, count)
            else:
                methods['sine_collocation'] = reconstruct_collocation(b, x)[0]
            for name, a in methods.items():
                rows.append(dict(layout=layout, sensors=count, method=name,
                    cap_distance_to_certified=float(cap_distance(a, period, reference, period)),
                    relative_l2=float(np.linalg.norm(cap_pad(a, reference) - reference)/np.linalg.norm(reference))))
    return pd.DataFrame(rows)


def cap_pad(a, like):
    out = np.zeros_like(like)
    k = min(a.shape[1], like.shape[1])
    out[:, :k] = a[:, :k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, default=HERE/'outputs_sensor')
    ap.add_argument('--sensors', type=int, nargs='+', default=[12, 16, 20, 24])
    ap.add_argument('--layouts', nargs='+', default=['regular', 'irregular'])
    ap.add_argument('--truth-modes', type=int, default=48)
    ap.add_argument('--true-modes', type=int, default=32, help='Galerkin size for true-PDE Newton test')
    ap.add_argument('--dt', type=float, default=0.1, help='sensor sampling interval')
    ap.add_argument('--burn', type=float, default=300.)
    ap.add_argument('--duration', type=float, default=3000.)
    ap.add_argument('--ridge', type=float, nargs='+', default=[0., 1e-10, 1e-8, 1e-6])
    ap.add_argument('--surrogate-dt', type=float, default=0.005)
    ap.add_argument('--rollout', type=float, default=2300.)
    ap.add_argument('--rollout-burn', type=float, default=300.)
    ap.add_argument('--harmonics', type=int, nargs='+', default=[16, 32])
    ap.add_argument('--spline-modes', type=int, nargs='+', default=[16, 32])
    ap.add_argument('--tries', type=int, default=8)
    ap.add_argument('--site-seed', type=int, default=0)
    args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    _, _, certified_full = certified_reference(0, 1)
    certified = certified_full[(len(certified_full) - 1)//2 - 48:(len(certified_full) - 1)//2 + 49]
    control = reconstruction_only(certified, CERTIFIED_PERIOD, args.layouts, args.sensors,
                                  args.spline_modes, args.true_modes, args.site_seed)
    control.to_csv(out/'reconstruction_only_control.csv', index=False)
    print(control.to_string(), flush=True)

    data, truth_config = simulate_truth(args.truth_modes, args.dt, args.burn, args.duration)
    records, results = [], []
    for layout in args.layouts:
        for count in args.sensors:
            record, rows = run_case(layout, count, data, truth_config, args, certified_full)
            print('CASE', json.dumps({k: v for k, v in record.items() if k != 'sites'}), flush=True)
            records.append(record); results.extend(rows)
    (out/'cases.json').write_text(json.dumps(records, indent=2) + '\n')
    pd.DataFrame(results).to_csv(out/'candidates.csv', index=False)
    (out/'config.json').write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                                               for k, v in vars(args).items()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
