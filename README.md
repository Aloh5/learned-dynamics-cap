# Validation of learned polynomial dynamics

> **Upload status:** The legacy DDSR benchmark has been removed and current equilibrium analysis moved to `analysis/`. The complete local `github_paper_export/` still needs to be uploaded: `notebooks/` and `results/` are not yet present on GitHub. The reproduction instructions below describe that complete export.

Code and numerical evidence for **Validation of Learned Polynomial Dynamics: Forecasts, Attractor Diagnostics, and Certified Periodic Orbits**, by Aaron Loh and J. D. Mireles James.

## Repository contents

| Path | Role in the paper |
| --- | --- |
| `notebooks/mackey_glass/` | Cubic forecasts, long-run diagnostics, degree-five surrogate orbit construction, and the fixed-parameter proof walkthrough |
| `notebooks/kuramoto_sivashinsky/` | Periodic-domain forecasts and relative-orbit pilot; separate Neumann proof walkthrough |
| `analysis/` | Current-model equilibrium searches and figure generation |
| `results/equilibria/` | Frozen fitted coefficients, equilibrium results, and source provenance for Sections 2.1–2.2 |
| `results/figures/` | Attractor comparisons and frozen candidate plots |
| `CAP/Mackey-Glass_PO/` | Fixed-parameter candidate, MATLAB/INTLAB proof, certificate, audit, and tests |
| `CAP/Kuramoto_PO/` | Neumann candidate construction, finite/tail scan, MATLAB/INTLAB proof, certificate, audit, and tests |

The periodic-domain KS numerical results and the Neumann KS theorem concern different boundary-value problems. Numerical diagnostics are not rigorous certificates of chaos or global model fidelity.

The manuscript, personal files, `archive/`, `paper/`, `overleaf_ready/`, scratch builds, abandoned DDSR predictor comparisons, pruning-sweep outputs, and logs are excluded. The source notebooks retain optional experiments for context; these are not additional claims of completed experiments in the paper.

## Environment and checks

Use Python 3.13. The core numerical versions are pinned to the recorded equilibrium run; the remaining packages use compatible ranges.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s analysis -p 'test_*.py' -v
python -m unittest discover -s CAP/Mackey-Glass_PO -p 'test_fixed_parameter.py' -v
python -m unittest discover -s CAP/Kuramoto_PO -p 'test_*.py' -v
```

These 19 tests check numerical derivatives, Fourier conversions, source identity and certificate inequalities. They do not replace interval validation.

## Reproduce the numerical results

Start JupyterLab from this repository root. The selected cubic MG forecast and long-run checks are in `Mackey_Glass_Current.ipynb`; surrogate orbit discovery and degree-five refinement are in `Mackey_Glass_CAP_Zero_Correction.ipynb`. `Kuramoto.ipynb` supplies the periodic KS forecast and relative-orbit pilot. Review the notebooks' run flags before executing expensive searches. Saved tables and small evidence files are included. Notebook execution output is cleared.

```sh
python analysis/run_equilibria.py
python analysis/generate_attractors.py
python analysis/generate_candidates.py
```

The equilibrium script refits the selected forecasting models and reruns the finite searches. The attractor script uses the saved models under `results/equilibria/`, with the paper's integration and projection protocols, and creates Figures 2 and 5. The candidate script plots the frozen approximations for Figure 6. Forecast Figures 1 and 4 and the long-run Figure 3 come from the forecasting notebooks; relevant saved images are in their output directories. Plot formatting may differ from the manuscript.

Original equilibrium metadata hashes are preserved as historical provenance. `export_sources_sha256` records the relocated export sources; relocation is not a scientific rerun. The historical package name in that provenance does not imply a DDSR dependency.

## Reproduce the interval certificates

MATLAB R2024b and INTLAB 13 were used for the saved certificates. Set the external installation paths:

```sh
INTLAB_DIR=/path/to/Intlab_V13 bash CAP/Mackey-Glass_PO/run_fixed_parameter_proof.sh
INTLAB_DIR=/path/to/Intlab_V13 bash CAP/Kuramoto_PO/run_surrogate_pde_proof.sh
```

`MATLAB_BINARY` and `PYTHON_BINARY` can override the executables. Drivers regenerate proof inputs and certificates; use a separate working copy if you want to preserve the frozen release evidence. Read each proof's Markdown audit for arithmetic assumptions and theorem scope.

The MG driver requires the included `notebooks/mackey_glass/outputs_zero_correction/MG_zero_correction_candidate.mat`. The KS driver regenerates `ks_neumann_fourier_proof_data.mat` (about 307 MB); this large inverse is deliberately excluded. Final fitted coefficients, continuation checkpoint, source trajectory and compact certificate evidence are included to support provenance and resume behavior.

No license is granted by this repository yet. Select a license if reuse permission is intended. Add the public repository URL and a release commit/tag to the manuscript when the release is public.
