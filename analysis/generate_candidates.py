"""Plot the frozen approximations used in Theorems 5.1 and 5.2 (Figure 6)."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cap-matplotlib')
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
mg = loadmat(ROOT / 'CAP/Mackey-Glass_PO/outputs_fixed_parameter/mg_fixed_parameter_candidate.mat')
u = mg['u'].ravel()
k = np.arange(-(len(u)//2), len(u)//2 + 1)
phase = np.linspace(0, 2*np.pi, 1001)
waveform = (np.exp(1j*phase[:, None]*k) @ u).real
mg_period = 20*np.pi / mg['omega'].item()
with np.load(ROOT / 'CAP/Kuramoto_PO/outputs_surrogate/ks_neumann_cap_candidate.npz') as ks:
    values = ks['velocity_time_values']
    length = ks['interval_length'].item()
    ks_period = ks['period'].item()
    points = np.linspace(0, length, 201)
    velocity = values @ np.sin(points[:, None]*np.pi/length*np.arange(1, values.shape[1]+1)).T
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
axes[0].plot(phase/(2*np.pi)*mg_period, waveform)
axes[0].set(xlabel='Physical time', ylabel='x(t)', title='Mackey–Glass frozen candidate')
plot = axes[1].pcolormesh(points, np.arange(len(values))/len(values)*ks_period, velocity, shading='auto', cmap='RdBu_r')
axes[1].set(xlabel='x', ylabel='Physical time', title='Neumann KS frozen candidate')
fig.colorbar(plot, ax=axes[1], label='Velocity u(x,t)')
out = ROOT / 'results/figures'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'certified_orbit_candidates.png', dpi=240)
plt.close(fig)
