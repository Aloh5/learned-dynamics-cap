"""Reproduce equilibrium benchmarks from the current forecasting notebook fits.

Run from the repository root: python analysis/run_equilibria.py
Only imports, configuration, solver/fitting definitions and training data are
executed. Forecast sweeps, orbit searches and CAPs are not executed.
"""
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
from equilibria import benchmark_mg, benchmark_ks, save_benchmark
import numpy as np


def load_notebook(relative, cells):
    path = ROOT / relative
    notebook = json.loads(path.read_text())
    namespace = {}
    for index, definitions_only in cells:
        source = "".join(notebook["cells"][index]["source"])
        tree = ast.parse(source)
        if definitions_only:
            tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, str(path) + f":cell{index}", "exec"), namespace)
    return namespace


def main():
    output = ROOT / "results/equilibria"
    output.mkdir(parents=True, exist_ok=True)
    mg_path = "notebooks/mackey_glass/Mackey_Glass_Current.ipynb"
    ks_path = "notebooks/kuramoto_sivashinsky/Kuramoto.ipynb"
    mg = load_notebook(mg_path, [(2, False), (4, False), (6, True), (8, False)])
    mg_model = mg["fit_sindy_model"](mg["X_delayed"], dt=mg["DT"], degree=mg["BASELINE_POLY_DEGREE"],
                                    alpha=mg["BASELINE_RIDGE_ALPHA"])
    print("Searching MG equilibria...", flush=True)
    save_benchmark(benchmark_mg(mg_model, mg["X_delayed"]), output)
    np.savez_compressed(output / "mackey_glass_model.npz", coefficients=mg_model.coefficients(),
                        powers=mg_model.feature_library.powers_, training=mg["X_delayed"])
    ks = load_notebook(ks_path, [(2, False), (4, False), (6, True), (10, True)])
    config = ks["CONFIG"]
    _, _, states = ks["simulate_ks_nodes"](ks["BASELINE_NODES"], config, seed=ks["BASELINE_SEED"])
    burn = round(config.burn_in / config.output_dt)
    count = round(config.train_time / config.output_dt) + 1
    training = states[burn:burn + count]
    ks_model = ks["fit_pysindy_model"](training, ks["BASELINE_DEGREE"], config)
    print("Searching periodic KS equilibria...", flush=True)
    save_benchmark(benchmark_ks(ks_model, training,
                   lambda x: ks["ks_rhs_spectral_state"](x, config.domain_length)), output)
    np.savez_compressed(output / "periodic_ks_model.npz", coefficients=ks_model.coefficients(),
                        powers=ks_model.feature_library.powers_, training=training)
    sources = [mg_path, ks_path, "analysis/run_equilibria.py", "analysis/equilibria.py"]
    metadata = {"sources": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pysindy", "scikit-learn"]},
                "scope": "Current selected MG cubic and periodic KS quadratic forecasting fits; no CAP models or comparison refits."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
