# Learned dynamics: forecasts and certified periodic orbits

Research by Aaron Loh and Dr. Jason D. Mireles-James (Florida Atlantic University).

A release-ready selection of code and small evidence files for the Mackey–Glass (MG) and Kuramoto–Sivashinsky (KS) study. This folder is a **standalone repository root**. The manuscript, personal working files, temporary builds, and large historical archive are outside it.

## What is here

| Path | Purpose |
| --- | --- |
| `notebooks/` | MG and periodic-KS forecasting notebooks, Neumann-KS and fixed-parameter MG proof notebooks, numerical tables and figures |
| `benchmark/` | Predictor comparison and equilibrium benchmark code plus comparison notebook |
| `results/` | Saved exploratory predictor comparison and equilibrium outputs |
| `CAP/Mackey-Glass_PO/` | Fixed-parameter MG candidate, certificate, driver, tests and proof audit |
| `CAP/Kuramoto_PO/` | Neumann-KS surrogate candidate, certificate, driver, tests and proof audit |

The periodic-domain KS forecast and Neumann-KS proof concern different boundary-value problems. The certificates are local existence statements for selected candidates; numerical forecasts and attractor diagnostics are separate results.

## Python setup

Use Python 3.13 and create a local environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The recorded equilibrium run used NumPy 2.4.4, SciPy 1.17.1, PySINDy 2.1.0, and scikit-learn 1.8.0; see `results/equilibria/metadata.json`. Other versions in `requirements.txt` are compatible ranges, not a byte-for-byte lockfile. The optional `ddeint` package is used only by a notebook comparison cell. Jupyter notebooks can be opened in JupyterLab from this repository root.

## Reproduce numerical checks

```sh
PYTHONPATH=benchmark/code python -m unittest ddsr_benchmark.test_paper_comparison ddsr_benchmark.test_equilibria -v
python -m unittest discover -s CAP/Mackey-Glass_PO -p 'test_fixed_parameter.py' -v
python -m unittest discover -s CAP/Kuramoto_PO -p 'test_*.py' -v
```

To inspect saved comparison results, open `benchmark/MG_KS_Predictor_Comparison.ipynb` with `RUN=False` (its default). To rerun the paper profile:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=benchmark/code python -m ddsr_benchmark.paper_comparison --profile paper --output results/predictor_comparison/paper
```

Run `python benchmark/run_equilibria.py` to rebuild the equilibrium results from the forecasting notebooks. The notebooks themselves regenerate the MG and periodic-KS figures and tables under their respective `outputs/` folders. Existing saved figures and tables are included for inspection. Their embedded execution outputs were cleared from the exported notebooks; the original working copies remain untouched.

## Reproduce proof data and certificates

The two proof drivers require MATLAB R2024b and INTLAB 13, in addition to the Python packages above. Set `INTLAB_DIR` to the INTLAB installation and optionally `MATLAB_BINARY` and `PYTHON_BINARY`:

```sh
INTLAB_DIR=/path/to/Intlab_V13 bash CAP/Mackey-Glass_PO/run_fixed_parameter_proof.sh
INTLAB_DIR=/path/to/Intlab_V13 bash CAP/Kuramoto_PO/run_surrogate_pde_proof.sh
```

The MG source waveform `notebooks/mackey_glass/outputs_zero_correction/MG_zero_correction_candidate.mat` is included because the fixed-parameter builder reads it. The KS driver regenerates `CAP/Kuramoto_PO/outputs_surrogate/ks_neumann_fourier_proof_data.mat`, a roughly 307 MB numerical inverse and scan file. It is intentionally excluded from Git because it exceeds GitHub's normal 100 MB per-file limit. The small frozen candidates and JSON certificates are included. See the proof-specific Markdown files for mathematical assumptions and interpretation.

The original metadata under `results/` retains hashes from the prior run. Each metadata file also has `export_source_sha256`, identifying the relocated source copies in this export. Relocation and notebook output clearing were not scientific reruns.

## Release scope

This export omits the manuscript, `overleaf_ready`, `paper`, the full `archive`, scratch data, Python bytecode, and the regenerable large inverse. It contains no license file; choose a license before a public release if reuse permission is intended.
