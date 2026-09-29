"""Reproducible MG/periodic-KS predictor comparison for the SIMODS draft.

This is a new numerical experiment, not a replacement for archived forecasts
or a CAP. Hyperparameters are chosen on validation rollouts only. See metadata
for differences in derivative access and initialization history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wasserstein_distance

from .dynamics import build_delay_embedding, simulate_mackey_glass, simulate_ks
from .library import polynomial_powers, transform_polynomial
from .models import fit_ddsr, fit_sindy_stlsq

ROOT = Path(__file__).resolve().parents[3]


def datasets():
    # Fine Euler reference, sampled every 0.1. This differs from older runs.
    _, fine = simulate_mackey_glass(dt=0.01, total_time=4500)
    mg = fine[::10]
    def embedded(a, b):
        return build_delay_embedding(mg, int(a * 10), int(b * 10), 45, 6)
    train = embedded(117, 417)
    yield dict(name="Mackey-Glass", dt=0.1, degree=3, tolerance=0.05,
               train=train, derivatives=np.gradient(train, 0.1, axis=0, edge_order=2),
               validation=embedded(700, 1300), test=embedded(1500, 2500),
               reference=embedded(3500, 4500), scalar=True,
               labels="finite differences of observations", boundary="delay=17",
               splits=[[117, 417], [700, 1300], [1500, 2500], [3500, 4500]],
               solver="Euler internal dt=0.01; observations dt=0.1")
    _, ks, dot = simulate_ks(total_time=650)
    # Gaps avoid sharing train/validation/test endpoints.
    yield dict(name="Periodic KS", dt=0.1, degree=2, tolerance=0.5,
               train=ks[1000:1700], derivatives=dot[1000:1700],
               validation=ks[2000:3200], test=ks[3500:4700], reference=ks[5000:6500],
               scalar=False, labels="exact spectral RHS of the 14-node system",
               boundary="periodic L=22, 14 nodes; NOT Neumann CAP",
               splits=[[100, 170], [200, 320], [350, 470], [500, 650]],
               solver="14-node spectral ETDRK4 internal dt=0.025; observations dt=0.1")


def ridge_features(features, target, alpha):
    scales = np.sqrt(np.mean(features ** 2, axis=0))
    scales[scales < 1e-14] = 1
    design = features / scales
    penalty = np.ones(design.shape[1]); penalty[0] = 0
    augmented = np.vstack([design, np.diag(np.sqrt(alpha * penalty))])
    rhs = np.vstack([target, np.zeros((design.shape[1], target.shape[1]))])
    coef = np.linalg.lstsq(augmented, rhs, rcond=None)[0] / scales[:, None]
    return coef


class Predictor:
    def __init__(self, kind, data, alpha=0, threshold=0, seed=0):
        self.kind, self.alpha, self.threshold, self.seed = kind, alpha, threshold, seed
        self.dt = data["dt"]
        x = data["train"]
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        self.dimension = x.shape[1]
        if kind in ("DPSR", "SINDy-STLSQ"):
            if kind == "DPSR":
                self.model = fit_ddsr(x, data["derivatives"], data["degree"], alpha)
            else:
                self.model = fit_sindy_stlsq(x, data["derivatives"], data["degree"], threshold, alpha)
        elif kind == "NVAR":
            self.powers = polynomial_powers(2 * self.dimension, 2)
            normalized = (x - self.mean) / self.scale
            taps = np.concatenate([normalized[1:-1], normalized[:-2]], axis=1)
            self.coef = ridge_features(transform_polynomial(taps, self.powers),
                                       normalized[2:] - normalized[1:-1], alpha)
        elif kind == "ESN":
            rng = np.random.default_rng(seed)
            self.W = rng.standard_normal((100, 100)) * (rng.random((100, 100)) < 0.1)
            self.W *= 0.9 / max(abs(np.linalg.eigvals(self.W)))
            self.Win = rng.uniform(-0.2, 0.2, (100, self.dimension + 1))
            self.leak = 0.3
            r = np.zeros((1, 100)); features = []
            normalized = (x - self.mean) / self.scale
            for value in normalized[:-1]:
                r = self.update(r, value[None])
                features.append(np.r_[1, value, r[0]])
            self.coef = ridge_features(np.asarray(features)[50:],
                                       (normalized[1:] - normalized[:-1])[50:], alpha)

    def update(self, r, values):
        inputs = np.column_stack([np.ones(len(values)), values])
        return (1 - self.leak) * r + self.leak * np.tanh(r @ self.W.T + inputs @ self.Win.T)

    def rollout(self, histories, steps, substeps=2):
        """Batch autonomous rollouts. Histories end at the forecast initial state."""
        initial = histories[:, -1].copy()
        output = np.full((len(initial), steps + 1, self.dimension), np.nan)
        output[:, 0] = initial
        alive = np.ones(len(initial), dtype=bool)
        current = initial.copy()
        if self.kind == "ESN":
            r = np.zeros((len(initial), 100))
            for j in range(histories.shape[1] - 1):
                r = self.update(r, (histories[:, j] - self.mean) / self.scale)
        previous = (histories[:, -2] - self.mean) / self.scale
        with np.errstate(over="ignore", invalid="ignore"):
            for step in range(steps):
                if not alive.any():
                    break
                if self.kind in ("DPSR", "SINDy-STLSQ"):
                    value = current.copy()
                    h = self.dt / substeps
                    for _ in range(substeps):
                        k1 = self.model.predict(value)
                        k2 = self.model.predict(value + h * k1 / 2)
                        k3 = self.model.predict(value + h * k2 / 2)
                        k4 = self.model.predict(value + h * k3)
                        value += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                elif self.kind == "Persistence":
                    value = current.copy()
                else:
                    normalized = (current - self.mean) / self.scale
                    if self.kind == "NVAR":
                        features = transform_polynomial(np.c_[normalized, previous], self.powers)
                    else:
                        r = self.update(r, normalized)
                        features = np.c_[np.ones(len(current)), normalized, r]
                    value = self.mean + self.scale * (normalized + features @ self.coef)
                    previous = normalized.copy()
                good = np.all(np.isfinite(value), axis=1) & np.all(abs((value-self.mean)/self.scale) < 20, axis=1)
                alive &= good
                output[alive, step + 1] = value[alive]
                # Dead rows are never revived; keep safe placeholders internally.
                current = np.where(alive[:, None], value, initial)
        return output

    def save(self, path):
        values = {k: v for k, v in vars(self).items() if k != "model"}
        if hasattr(self, "model"):
            values.update(coefficients=self.model.coefficients, powers=self.model.powers)
        np.savez_compressed(path, **values)


def histories_and_truth(data, window, duration, count):
    steps = int(round(duration / data["dt"]))
    starts = np.linspace(50, len(window) - steps - 1, count, dtype=int)
    if len(set(starts)) != count or min(starts) < 50:
        raise ValueError("Window too short for requested initializations")
    histories = np.stack([window[i-50:i+1] for i in starts])
    truth = np.stack([window[i:i+steps+1] for i in starts])
    return histories, truth, starts


def score(data, prediction, truth, normalization):
    rows = []
    for i, (pred, true) in enumerate(zip(prediction, truth)):
        completed = bool(np.isfinite(pred).all())
        error = (pred[:, :1] - true[:, :1]) ** 2 if data["scalar"] else (pred-true) ** 2
        mse = error.mean(axis=1)
        crossings = np.flatnonzero(np.cumsum(mse) * data["dt"] > data["tolerance"])
        # Failed rollouts stay in the denominator, with zero horizon.
        horizon = (crossings[0] if len(crossings) else len(pred)-1) * data["dt"] if completed else 0.
        rows.append(dict(case=i, completed=completed, horizon=horizon,
                         nrmse=float(np.sqrt(mse.mean()) / normalization) if completed else np.nan))
    return rows


def diagnostics(data, prediction):
    reference = data["reference"]
    if data["scalar"]:
        reference = reference[:, :1]
    rows = []
    for i, pred in enumerate(prediction):
        completed = bool(np.isfinite(pred).all())
        row = dict(case=i, long_completed=completed, wasserstein=np.nan,
                   std_ratio=np.nan, spectrum_l1=np.nan)
        if completed:
            sample = pred[len(pred)//4:]
            if data["scalar"]:
                sample = sample[:, :1]
            nseg = min(256, len(sample), len(reference))
            def spectrum(values):
                p = welch(values, fs=1/data["dt"], nperseg=nseg, axis=0)[1].mean(axis=1)[1:]
                return p / max(p.sum(), 1e-30)
            norm = max(float(reference.std()), 1e-12)
            row.update(wasserstein=float(wasserstein_distance(sample.ravel(), reference.ravel()) / norm),
                       std_ratio=float(sample.std()/norm),
                       spectrum_l1=float(np.sum(abs(spectrum(sample)-spectrum(reference)))))
        rows.append(row)
    return rows


def run(output, profile="pilot"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(dict(profile=profile, complete=False))+"\n")
    count = 2 if profile == "smoke" else (6 if profile == "paper" else 3)
    seeds = [0, 1, 2] if profile == "paper" else [0]
    alphas = [0., 1e-6, 1e-4, 1e-2] if profile != "smoke" else [0., 1e-4]
    thresholds = [1e-4, 1e-2, 1e-1] if profile != "smoke" else [1e-2]
    all_cases, sweep, summaries, all_long, all_checks, metadata = [], [], [], [], [], []
    for data in datasets():
        print("DATA", data["name"], flush=True)
        duration = (150 if data["scalar"] else 30) if profile == "paper" else (40 if data["scalar"] else 15)
        long_duration = (1000 if data["scalar"] else 200) if profile == "paper" else (200 if data["scalar"] else 60)
        validation_history, validation_truth, _ = histories_and_truth(data, data["validation"], duration, count)
        test_history, test_truth, test_starts = histories_and_truth(data, data["test"], duration, count)
        norm = float(data["train"][:, 0].std() if data["scalar"] else data["train"].std())
        slug = data["name"].lower().replace(" ", "_").replace("-", "_")
        np.savez_compressed(output / f"{slug}_data.npz", **{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
        metadata.append({k:v for k,v in data.items() if not isinstance(v,np.ndarray)} | dict(duration=duration, long_duration=long_duration, test_start_indices=test_starts.tolist(), normalization=norm))
        for kind in ["Persistence", "DPSR", "SINDy-STLSQ", "NVAR", "ESN"]:
            configs = [(0, 0, 0)] if kind == "Persistence" else [(a, t, s) for a in alphas for t in (thresholds if kind == "SINDy-STLSQ" else [0]) for s in (seeds if kind == "ESN" else [0])]
            candidates = []
            for alpha, threshold, seed in configs:
                begin = time.perf_counter()
                model = Predictor(kind, data, alpha, threshold, seed)
                fit_seconds = time.perf_counter()-begin
                pred = model.rollout(validation_history, len(validation_truth[0])-1)
                records = score(data, pred, validation_truth, norm)
                table = pd.DataFrame(records)
                finite_error = table.nrmse.mean()
                row = dict(system=data["name"], family=kind, alpha=alpha, threshold=threshold, seed=seed,
                           validation_completed=float(table.completed.mean()), validation_horizon=float(table.horizon.mean()),
                           validation_nrmse=float(finite_error), fit_seconds=fit_seconds)
                sweep.append(row)
                key = (-row["validation_completed"], -row["validation_horizon"], finite_error if np.isfinite(finite_error) else np.inf)
                candidates.append((key, model, row))
            _, chosen, selected = min(candidates, key=lambda entry: entry[0])
            begin = time.perf_counter()
            prediction = chosen.rollout(test_history, len(test_truth[0])-1)
            seconds = time.perf_counter()-begin
            records = score(data, prediction, test_truth, norm)
            all_cases.extend(dict(system=data["name"], family=kind, **r) for r in records)
            long_prediction = chosen.rollout(test_history, int(round(long_duration/data["dt"])))
            long_records = diagnostics(data, long_prediction)
            all_long.extend(dict(system=data["name"], family=kind, **r) for r in long_records)
            # Numerical sensitivity of the selected continuous-time predictors.
            if kind in ("DPSR", "SINDy-STLSQ"):
                finer = chosen.rollout(test_history, len(test_truth[0])-1, substeps=4)
                fine_records = score(data, finer, test_truth, norm)
                for coarse, fine in zip(records, fine_records):
                    all_checks.append(dict(system=data["name"], family=kind, case=fine["case"],
                                           horizon_dt_half=coarse["horizon"], horizon_dt_quarter=fine["horizon"],
                                           completed_dt_half=coarse["completed"], completed_dt_quarter=fine["completed"]))
            table, long_table = pd.DataFrame(records), pd.DataFrame(long_records)
            summary = dict(**selected, test_completed=float(table.completed.mean()),
                           test_horizon=float(table.horizon.mean()), test_nrmse=float(table.nrmse.mean()),
                           long_completed=float(long_table.long_completed.mean()),
                           wasserstein=float(long_table.wasserstein.mean()), std_ratio=float(long_table.std_ratio.mean()),
                           spectrum_l1=float(long_table.spectrum_l1.mean()), rollout_seconds=seconds)
            summaries.append(summary)
            chosen.save(output / f"{slug}_{kind.lower()}_model.npz")
            np.savez_compressed(output / f"{slug}_{kind.lower()}_rollouts.npz", prediction=prediction, truth=test_truth,
                                long_prediction=long_prediction, reference=data["reference"], dt=data["dt"])
            print("RESULT", data["name"], kind, summary["test_horizon"], summary["test_completed"], flush=True)
            # Incremental outputs survive interruption; completion recorded below.
            pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
            pd.DataFrame(sweep).to_csv(output / "validation_sweep.csv", index=False)
    for filename, records in [("forecast_cases.csv",all_cases),("long_run_cases.csv",all_long),("integration_checks.csv",all_checks)]:
        pd.DataFrame(records).to_csv(output / filename, index=False)
    run_metadata = dict(profile=profile, complete=True, datasets=metadata, alphas=alphas, thresholds=thresholds, esn_seeds=seeds,
                        selection="validation completion fraction, mean failure-inclusive horizon, completed-rollout NRMSE; stable grid-order ties",
                        failure="nonfinite or >20 training standard deviations in any state; zero forecast horizon, retained failure count; NRMSE/statistics conditional on completion",
                        initialization="51 observed states ending at start available; ESN warmup 50 preceding states, NVAR previous/current, vector fields current; no future truth during rollout",
                        access="DPSR/STLSQ share derivative labels; NVAR/ESN/persistence use values only. KS derivative access is privileged.",
                        scope="new finite-reference pilot; not a reproduction of archived forecasts, not Neumann KS, not a CAP or a state-of-the-art ranking",
                        independence="initializations share one deterministic reference trajectory; no independent-replicate confidence intervals",
                        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name("models.py"),Path(__file__).with_name("dynamics.py"),Path(__file__).with_name("library.py")]})
    (output / "metadata.json").write_text(json.dumps(run_metadata, indent=2)+"\n")
    make_figures(output)
    return pd.DataFrame(summaries)


def make_figures(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = pd.read_csv(output / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, (name, group) in zip(axes, table.groupby("system", sort=False)):
        ax.bar(group.family, group.test_horizon, color=["#9ca3af", "#155e75", "#0891b2", "#d97706", "#7c3aed"])
        ax.set(title=name, ylabel="Mean forecast horizon (physical time)")
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output / "forecast_comparison.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "pilot", "paper"], default="pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "results/predictor_comparison/pilot")
    args = parser.parse_args()
    run(args.output, args.profile)
