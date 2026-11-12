# Learned dynamics and computer-assisted proofs

Dense polynomial surrogate reconstruction for Mackey–Glass (MG) and
Kuramoto–Sivashinsky (KS), followed by forecasting, numerical dynamics, and
computer-assisted validation of selected periodic orbits.

## Start here

Install from the repository root with Python 3.10 or newer:

```sh
python -m pip install -e ".[notebooks]"
jupyter lab
```

No PySINDy dependency is required. Open notebooks from the repository root or
their own directory. Each forecasting notebook fits and evaluates its baseline
by default; sweeps, pruning, long diagnostics, equilibrium searches and orbit
construction are explicitly enabled in its configuration cell.

| Notebook | Purpose | Manuscript connection |
| --- | --- | --- |
| [MG forecasting](notebooks/mackey_glass/mg_delay_forecasting.ipynb) | Delay-coordinate ridge fitting and held-out forecasts | Polynomial regression; MG prediction and local dynamics |
| [MG orbit construction](notebooks/mackey_glass/mg_surrogate_orbit_construction.ipynb) | Optional surrogate-only discovery and higher-degree refinement | MG periodic-candidate construction |
| [MG certification](notebooks/mackey_glass/mg_periodic_orbit_certification.ipynb) | Inspect the frozen fixed-parameter candidate and certificate | Fixed-parameter MG proof |
| [Periodic KS forecasting](notebooks/kuramoto_sivashinsky/ks_periodic_forecasting.ipynb) | Dense quadratic nodal fitting and held-out forecasts | Spatial regression; periodic KS assessment |
| [Neumann KS certification](notebooks/kuramoto_sivashinsky/ks_neumann_orbit_certification.ipynb) | Surrogate-only candidate construction and PDE validation | Neumann KS construction and certificate |

## Method and structure

The baseline method fits a complete polynomial library without iterative
sparsity thresholding. MG uses finite-difference derivative labels and
RMS-normalized degree-weighted ridge. Periodic KS uses exact spectral labels
on trajectory snapshots and unscaled, unregularized least squares. Neumann KS
uses exact Galerkin labels on paired independent random states, a parity-split
quadratic fit, and reflection symmetrization. These sampling and boundary-value
problems support distinct scientific claims.

| Path | Contents |
| --- | --- |
| `dpsr/` | Shared regression, reference solvers, polynomial models, orbit helpers, metrics, and equilibrium benchmarks |
| `notebooks/` | Five purpose-named experiment and certification entry points |
| `CAP/` | Proof drivers, frozen candidates, certificates, and audits |
| `docs/` | Regression methodology and numerical conventions |
| `analysis/` | Equilibrium reproduction and stored-model figure generation |
| `results/` | Stored numerical evidence and figures |
| `tests/` | Shared regression and equilibrium checks |
| `outputs/` | Ignored, regenerable experiment outputs |

See [methodology](docs/methodology.md) for data regimes and regression
conventions. To regenerate numerical equilibrium results, run
`python analysis/run_equilibria.py`. Figure scripts in `analysis/` use the
stored models in `results/equilibria/`.

## Verification and proofs

For the tested scientific package versions, see `requirements.txt`.

```sh
python -m unittest discover -s tests -v
python scripts/verify_notebooks.py
(cd CAP/Kuramoto_PO && python -m unittest test_ks_neumann test_surrogate_neumann -v)
(cd CAP/Mackey-Glass_PO && python -m unittest test_fixed_parameter -v)
```

Numerical checks do not replace interval validation. Reproducing the rigorous
proof requires MATLAB and INTLAB; see the
[MG proof documentation](CAP/Mackey-Glass_PO/FIXED_PARAMETER_PROOF.md) and
[Neumann KS proof documentation](CAP/Kuramoto_PO/README.md). Proof execution is
opt-in. Default notebook runs inspect certificates without regenerating them.
MG construction writes new experiments under `outputs/`; the fixed-parameter
proof continues to use its documented frozen candidate in
`notebooks/mackey_glass/outputs_zero_correction/`.

## Paper

"Data-Driven Discovery and Rigorous Certification of Periodic Solutions in
Infinite-Dimensional Dynamical Systems", Aaron Loh and J. D. Mireles James.
arXiv link: TBA.

## Reproducing the theorems

Paths below are relative to the repository root. Run shell entry points with `bash`.

| Paper result | Entry point |
|---|---|
| Theorem 6.1 | `CAP/Mackey-Glass_PO/run_fixed_parameter_proof.sh` |
| Theorem 6.2 | `CAP/Mackey-Glass_PO/run_newton_refined_proof.sh` |
| Theorem 6.3 | `CAP/Kuramoto_PO/run_surrogate_pde_proof.sh` |
| Theorem 6.4 | `CAP/Kuramoto_PO/run_newton_refined_proof.sh` |
| Theorem 6.5 | `CAP/Kuramoto_PO/run_sensor_refined_proof.sh` |
| Table 7 / Figure 6 | `CAP/Kuramoto_PO/sensor_convergence.py` and `CAP/Kuramoto_PO/summarize_sensor_convergence.py` |

### INTLAB configuration

Set `INTLAB_PATH` to your installed INTLAB directory before running a proof or
an opt-in certification notebook. For example:

```sh
export INTLAB_PATH="/path/to/INTLAB"
export MATLAB_BINARY="/path/to/matlab"
bash CAP/Mackey-Glass_PO/run_newton_refined_proof.sh
```

MATLAB and INTLAB are separately installed dependencies. Python tests do not
replace interval validation. See `CAP/Kuramoto_PO/NEWTON_REFINEMENT.md` for
the transfer of the original full-space operator bounds.

### Station/sensor evidence

| Path | Contents |
|---|---|
| `CAP/Kuramoto_PO/sensor_experiment.py` | Station observations, surrogate fitting, reconstruction and diagnostics |
| `CAP/Kuramoto_PO/outputs_sensor/` | Initial station experiment CSV/JSON |
| `CAP/Kuramoto_PO/outputs_sensor_convergence/` | Table 7 summary, per-count evidence and Figure 6 (`ks_sensor_convergence.png`) |
| `CAP/Kuramoto_PO/outputs_sensor_proof/` | Frozen station candidate, continuation history and certificate |

Run `python CAP/Kuramoto_PO/summarize_sensor_convergence.py` to rebuild the
summary and figure from the included small artifacts. `python CAP/Kuramoto_PO/sensor_convergence.py --help` describes the per-count simulation options. Its `truth.npz` cache is
regenerated automatically and is intentionally excluded.

Large `*fourier_proof_data.mat` matrices are intentionally excluded. Before
KS refinement or its matrix-dependent numerical tests, regenerate the original
matrix with:

```sh
python CAP/Kuramoto_PO/scan_block_proof.py --candidate CAP/Kuramoto_PO/outputs_surrogate/ks_neumann_cap_candidate.mat --output CAP/Kuramoto_PO/outputs_surrogate/ks_neumann_fourier_proof_data.mat
```

The sensor proof shell script regenerates its own matrix. Historical refinement
manifests preserve the hashes of the executed proof dependencies; packaging
changes to paths and documentation are recorded separately and do not assert
that MATLAB/INTLAB was rerun.
