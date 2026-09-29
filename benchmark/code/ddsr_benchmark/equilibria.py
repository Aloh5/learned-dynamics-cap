"""Numerical equilibrium benchmarks for the manuscript's forecasting models.

Search budgets are fixed by start count, never wall time. These are numerical
root searches, not completeness proofs. No coefficients are refitted here.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment, root
from scipy.special import lambertw

RESIDUAL_TOL = 1e-8
STABILITY_TOL = 1e-5


class PolynomialField:
    def __init__(self, model):
        self.coefficients = np.asarray(model.coefficients())
        self.powers = np.asarray(model.feature_library.powers_, dtype=int)
        self.derivative_powers = []
        for j in range(self.powers.shape[1]):
            powers = self.powers.copy()
            powers[:, j] = np.maximum(0, powers[:, j] - 1)
            self.derivative_powers.append(powers)

    def __call__(self, x):
        return self.coefficients @ np.prod(x ** self.powers, axis=1)

    def jacobian(self, x):
        features = np.column_stack([
            self.powers[:, j] * np.prod(x ** powers, axis=1)
            for j, powers in enumerate(self.derivative_powers)
        ])
        return self.coefficients @ features


def centered_jacobian(function, x, step=1e-6):
    eye = step * np.eye(len(x))
    return np.column_stack([(function(x + e) - function(x - e)) / (2 * step)
                            for e in eye])


def shift_distance(a, b):
    return min(float(np.linalg.norm(np.roll(a, k) - b)) for k in range(len(a)))


def stability(values):
    real = np.asarray(values).real
    return {"stable": int(np.sum(real < -STABILITY_TOL)),
            "unstable": int(np.sum(real > STABILITY_TOL)),
            "neutral": int(np.sum(np.abs(real) <= STABILITY_TOL))}


def complex_pairs(values):
    values = sorted(values, key=lambda z: (-z.real, -z.imag))
    return [[float(z.real), float(z.imag)] for z in values]


def search(field, jacobian, starts, *, mean=None, periodic=False):
    roots, attempts = [], []
    distance = shift_distance if periodic else lambda a, b: np.linalg.norm(a - b)
    duplicate_tol = 1e-3 if periodic else 1e-5
    for index, start in enumerate(starts):
        if mean is None:
            sol = root(field, start, jac=jacobian, options={"xtol": 1e-10, "maxfev": 1000})
        else:
            def fun(x):
                return np.r_[field(x), 10 * (np.mean(x) - mean)]

            def jac(x):
                return np.vstack([jacobian(x), np.full(len(x), 10 / len(x))])

            sol = least_squares(fun, start, jac=jac, xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=500)
        residual = float(np.linalg.norm(field(sol.x)))
        mean_error = 0.0 if mean is None else float(abs(np.mean(sol.x) - mean))
        accepted = bool(sol.success and np.all(np.isfinite(sol.x))
                        and residual < RESIDUAL_TOL and mean_error < RESIDUAL_TOL)
        attempts.append({"start": index, "solver_success": bool(sol.success),
                         "residual": residual, "mean_error": mean_error,
                         "accepted": accepted, "nfev": int(sol.nfev)})
        if accepted and all(distance(sol.x, previous) >= duplicate_tol for previous in roots):
            roots.append(sol.x.copy())
    return roots, attempts


def root_record(x, field, jacobian):
    eigenvalues = np.linalg.eigvals(jacobian(x))
    return {"state": x.tolist(), "residual": float(np.linalg.norm(field(x))),
            "eigenvalues": complex_pairs(eigenvalues), "stability": stability(eigenvalues),
            "jacobian_fd_error": float(np.max(np.abs(jacobian(x) - centered_jacobian(field, x))))}


def add_attractor_distances(records, training, keep=None):
    """Distance to sampled training states, not to the full mathematical attractor.

    The archived MG experiment selected the single nearest root (n_keep=1),
    rather than imposing an independently chosen distance threshold.
    """
    for record in records:
        record["min_distance_to_training"] = float(np.min(
            np.linalg.norm(training - np.asarray(record["state"]), axis=1)))
    order = sorted(range(len(records)), key=lambda i: records[i]["min_distance_to_training"])
    for rank, index in enumerate(order, 1):
        records[index]["attractor_distance_rank"] = rank
        if keep is not None:
            records[index]["retained_by_historical_filter"] = rank <= keep


def dde_spectrum(equilibrium, branches=64):
    # lambda + gamma = a exp(-lambda*tau), hence
    # lambda = W_k(a*tau*exp(gamma*tau))/tau - gamma.
    # W definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lambertw.html
    beta, gamma, tau, exponent = 0.2, 0.1, 17.0, 10
    a = beta * (1 + (1 - exponent) * equilibrium**exponent) / (1 + equilibrium**exponent)**2
    values = lambertw(a * tau * np.exp(gamma * tau), np.arange(-branches, branches + 1)) / tau - gamma
    residual = np.max(np.abs(values + gamma - a * np.exp(-values * tau)))
    return values, float(residual)


def benchmark_mg(model, training, n_random=256, seed=42):
    field = PolynomialField(model)
    constants = np.array([-1.0, 0.0, 1.0])
    references = [np.full(training.shape[1], c) for c in constants]
    starts = references + list(training[np.linspace(0, len(training) - 1, 20, dtype=int)])
    starts += list(np.random.default_rng(seed).uniform(0, 2, (n_random, training.shape[1])))
    roots, attempts = search(field, field.jacobian, starts)
    records = []
    for x in sorted(roots, key=lambda z: np.linalg.norm(z - references[-1])):
        record = root_record(x, field, field.jacobian)
        index = int(np.argmin([np.linalg.norm(x - ref) for ref in references]))
        record.update(nearest_true_constant=float(constants[index]),
                      position_error=float(np.linalg.norm(x - references[index])),
                      coordinate_spread=float(np.ptp(x)))
        records.append(record)
    add_attractor_distances(records, training, keep=1)
    reference_records = []
    for c, ref in zip(constants, references):
        spectrum, residual = dde_spectrum(c)
        wider, _ = dde_spectrum(c, branches=128)
        assert np.allclose(np.sort_complex(spectrum[spectrum.real > 0]),
                           np.sort_complex(wider[wider.real > 0]))
        reference_records.append({"constant": float(c), "eigenvalues": complex_pairs(spectrum),
                                  "unstable_roots_in_branch_window": stability(spectrum)["unstable"],
                                  "characteristic_residual": residual,
                                  "surrogate_residual_at_exact_equilibrium": float(np.linalg.norm(field(ref))),
                                  "nearest_learned_distance": min((float(np.linalg.norm(ref-x)) for x in roots), default=None)})
    return {"system": "mackey_glass", "model": "selected single-window cubic forecast model",
            "seed": seed, "n_random": n_random, "random_bounds": [0, 2],
            "residual_tolerance": RESIDUAL_TOL, "stability_tolerance": STABILITY_TOL,
            "dde_branch_window": [-64, 64], "roots": records, "references": reference_records,
            "attractor_filter": {"keep": 1, "metric": "minimum Euclidean distance to training embedding",
                                 "source": "archive/mackey_glass/MG_FINAL.ipynb cell 39",
                                 "interpretation": "ranking heuristic, not a spurious-root or attractor-membership proof"},
            "attempts": attempts, "exhaustive": False}


def benchmark_ks(model, training, true_rhs, n_random=128, seed=42):
    field = PolynomialField(model)
    mean = float(np.mean(training[0]))
    # The reference field is quadratic, so centered differences with step 1
    # differentiate it exactly in exact arithmetic, avoiding small-step cancellation.
    true_jac = lambda x: centered_jacobian(true_rhs, x, step=1.0)
    starts = [np.full(training.shape[1], mean)]
    starts += list(training[np.linspace(0, len(training) - 1, 20, dtype=int)])
    starts += list(np.random.default_rng(seed).uniform(training.min(), training.max(),
                                                     (n_random, training.shape[1])))
    learned, learned_attempts = search(field, field.jacobian, starts, mean=mean, periodic=True)
    truth, true_attempts = search(true_rhs, true_jac, starts, mean=mean, periodic=True)
    learned.sort(key=np.linalg.norm)
    truth.sort(key=np.linalg.norm)
    learned_records = [root_record(x, field, field.jacobian) for x in learned]
    true_records = [root_record(x, true_rhs, true_jac) for x in truth]
    add_attractor_distances(learned_records, training)
    add_attractor_distances(true_records, training)
    # One-to-one root assignment prevents several learned roots from all being
    # counted as recovering a single reference root. Gate matches at 1e-3.
    matches = []
    if learned and truth:
        cost = np.array([[shift_distance(x, y) for y in truth] for x in learned])
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e-3:
                continue
            ev_l = np.linalg.eigvals(field.jacobian(learned[i]))
            ev_t = np.linalg.eigvals(true_jac(truth[j]))
            a, b = linear_sum_assignment(abs(ev_l[:, None] - ev_t[None, :]))
            matches.append({"learned_index": int(i), "true_index": int(j),
                            "position_error": float(cost[i, j]),
                            "true_residual_at_learned_root": float(np.linalg.norm(true_rhs(learned[i]))),
                            "max_matched_eigenvalue_error": float(np.max(abs(ev_l[a] - ev_t[b]))),
                            "stability_matches": stability(ev_l) == stability(ev_t)})
    return {"system": "periodic_ks", "model": "14-node quadratic forecast model",
            "seed": seed, "n_random": n_random, "target_mean": mean,
            "random_bounds": [float(training.min()), float(training.max())],
            "residual_tolerance": RESIDUAL_TOL, "stability_tolerance": STABILITY_TOL,
            "match_tolerance": 1e-3, "roots": learned_records, "references": true_records,
            "matches": matches, "learned_attempts": learned_attempts,
            "true_attempts": true_attempts, "exhaustive": False}


def save_benchmark(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{result['system']}_equilibria.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['system']}: {len(result['roots'])} learned roots; results: {path}")
    return path


def summary_rows(result):
    """Compact notebook display; complete coordinates/spectra stay in JSON."""
    if result["system"] == "mackey_glass":
        return [{"root": i, "nearest_true_constant": r["nearest_true_constant"],
                 "min_distance_to_training": r["min_distance_to_training"],
                 "retained_by_historical_filter": r["retained_by_historical_filter"],
                 "position_error": r["position_error"], "coordinate_spread": r["coordinate_spread"],
                 "residual": r["residual"], **r["stability"]}
                for i, r in enumerate(result["roots"])]
    return [{**m, "min_distance_to_training": result["roots"][m["learned_index"]]["min_distance_to_training"],
             **result["roots"][m["learned_index"]]["stability"]} for m in result["matches"]]
