"""Tables and figure for the sensor-count study (sensor_convergence.py).

Per sensor count M the reported candidate is the one whose true Newton limit
is the T ~ 75.72 orbit certified in outputs_sensor_proof (present for most M);
otherwise the candidate closest to its own Newton limit. The reconstruction-
only control samples that certified orbit at the M sites and applies the DST,
which isolates the error of the reconstruction from that of the surrogate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sensor_experiment import HERE, sensor_sites, sine_matrix, reconstruct_dst, cap_distance

OUT = HERE/'outputs_sensor_convergence'
FIGURE = OUT/'ks_sensor_convergence.png'
TARGET_PERIOD = 75.7239
SENSORS = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36, 40]


def load(order):
    rows, histories, candidates = [], {}, {}
    for M in SENSORS:
        path = OUT/f'fd{order}'/f'M{M}.json'
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        row = dict(sensors=M, fd_order=order, status=record['status'],
                   best_heldout_error=record['best_heldout_error'],
                   heldout_error=record.get('heldout_error'), ridge=record.get('ridge'),
                   orbits_found=len(record.get('orbits', [])))
        converged = [o for o in record.get('orbits', []) if o['newton_converged']]
        if converged:
            same = [o for o in converged if abs(o['true_period'] - TARGET_PERIOD) < 1e-2]
            best = min(same or converged, key=lambda o: o['distance_to_true_orbit'])
            row.update({k: best[k] for k in ('surrogate_period', 'true_period',
                        'initial_defect_max', 'initial_defect_oversampled_rms',
                        'first_newton_correction', 'newton_corrections',
                        'distance_to_true_orbit')})
            row['true_periods_found'] = sorted({round(o['true_period'], 3) for o in converged})
            histories[M] = best['newton_history']
            with np.load(OUT/f'fd{order}'/f'M{M}_candidates.npz') as z:
                candidates[M] = (z[f"candidate_{best['orbit']}"], z[f"refined_{best['orbit']}"])
        rows.append(row)
    return pd.DataFrame(rows), histories, candidates


def control(reference, period):
    rows = []
    for M in SENSORS + [44, 48]:
        x = sensor_sites('regular', M)
        a = reconstruct_dst(reference @ sine_matrix(x, reference.shape[1]).T, M)
        rows.append(dict(sensors=M, distance=float(cap_distance(a, period, reference, period))))
    return pd.DataFrame(rows)


def main():
    certified = loadmat(HERE/'outputs_sensor_proof'/'ks_neumann_cap_candidate.mat')
    reference = certified['velocity_time_fourier']
    period = float(certified['period'].item())
    tables = {order: load(order) for order in (2, 4)}
    summary = pd.concat([tables[o][0] for o in (2, 4)], ignore_index=True)
    summary.to_csv(OUT/'summary.csv', index=False)
    recon = control(reference, period)
    recon.to_csv(OUT/'reconstruction_control.csv', index=False)
    pd.set_option('display.width', 220)
    print(summary.drop(columns=['true_periods_found'], errors='ignore').to_string())
    print(recon.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for order, marker in ((2, 'o'), (4, 's')):
        df = tables[order][0]
        ax.semilogy(df.sensors, df.best_heldout_error, marker + '-', color='C0',
                    mfc='none' if order == 4 else 'C0', label=f'held-out $\\dot Y$ error (FD{order})')
        ok = df.dropna(subset=['distance_to_true_orbit'])
        ax.semilogy(ok.sensors, ok.initial_defect_max, marker + '-', color='C1',
                    mfc='none' if order == 4 else 'C1', label=f'initial defect (FD{order})')
        ax.semilogy(ok.sensors, ok.distance_to_true_orbit, marker + '-', color='C2',
                    mfc='none' if order == 4 else 'C2', label=f'distance in (25) (FD{order})')
        bad = df[df.status != 'ok']
        if order == 2 and len(bad):
            ax.semilogy(bad.sensors, np.full(len(bad), 3e2), 'kx', ms=8, label='rollout diverged')
    ax.semilogy(recon.sensors, recon.distance, 'k--', lw=1, label='exact orbit + DST only')
    ax.axhline(1e-5, color='r', lw=0.8, ls=':', label='zero-Newton proof radius')
    ax.set_xlabel('sensors $M$'); ax.set_ylabel('$\\log_{10}$ scale')
    ax.set_title('(a) accuracy versus sensor count'); ax.legend(fontsize=6.5, loc='lower left')

    ax = axes[1]
    histories = tables[4][1]
    for i, (M, history) in enumerate(sorted(histories.items())):
        ax.semilogy([h['iteration'] for h in history], [h['residual_max'] for h in history],
                    '.-', color=plt.cm.viridis(i/max(1, len(histories) - 1)), label=f'M={M}')
    ax.axhline(1e-10, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('true-equation Newton iteration'); ax.set_ylabel('max collocation residual')
    ax.set_title('(b) true-equation Newton iterations (FD4)'); ax.legend(fontsize=6.5, ncol=2)

    ax = axes[2]
    candidates = tables[4][2]
    for i, M in enumerate((16, 20, 24)):
        if M not in candidates:
            continue
        candidate, refined = candidates[M]
        frozen = np.zeros_like(refined)
        frozen[:, :candidate.shape[1]] = candidate
        k = np.arange(1, refined.shape[1] + 1)
        color = plt.cm.viridis(i/2)
        ax.semilogy(k, np.max(np.abs(refined), axis=0), '-', color=color, lw=1.2,
                    label=f'true orbit, M={M}')
        ax.semilogy(k, np.maximum(np.max(np.abs(frozen - refined), axis=0), 1e-17), '--',
                    color=color, lw=1, label=f'candidate error, M={M}')
    ax.axhline(np.finfo(float).eps, color='gray', lw=0.8, ls=':', label='machine precision')
    ax.set_xlabel('sine mode $k$'); ax.set_ylabel('$\\max_n |a_{nk}|$')
    ax.set_title('(c) coefficients and candidate error (FD4)'); ax.legend(fontsize=6.5)
    fig.tight_layout()
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, dpi=200)
    print('wrote', FIGURE)


if __name__ == '__main__':
    main()
