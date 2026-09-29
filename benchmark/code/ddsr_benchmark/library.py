from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np


@lru_cache(maxsize=64)
def polynomial_powers(n_state: int, degree: int) -> np.ndarray:
    """Return all monomial exponents through total ``degree``, bias first."""
    powers = [np.zeros(n_state, dtype=int)]
    for total_degree in range(1, degree + 1):
        for indices in combinations_with_replacement(range(n_state), total_degree):
            exponent = np.zeros(n_state, dtype=int)
            for index in indices:
                exponent[index] += 1
            powers.append(exponent)
    result = np.asarray(powers, dtype=int)
    result.setflags(write=False)
    return result


def transform_polynomial(states: np.ndarray, powers: np.ndarray) -> np.ndarray:
    states = np.atleast_2d(np.asarray(states, dtype=float))
    return np.prod(states[:, None, :] ** powers[None, :, :], axis=2)


def feature_names(powers: np.ndarray) -> list[str]:
    names: list[str] = []
    for power in powers:
        factors = []
        for index, exponent in enumerate(power):
            if exponent == 1:
                factors.append(f"x{index}")
            elif exponent > 1:
                factors.append(f"x{index}^{exponent}")
        names.append(" ".join(factors) if factors else "1")
    return names

