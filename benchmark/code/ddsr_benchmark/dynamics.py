from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def build_delay_embedding(series: np.ndarray, start: int, stop: int, lag: int, dimension: int) -> np.ndarray:
    if start - (dimension - 1) * lag < 0:
        raise ValueError("Delay embedding reaches before the available time series")
    return np.column_stack([series[start - j * lag : stop - j * lag] for j in range(dimension)])


def lorenz_rhs(_time: float, state: np.ndarray, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def simulate_lorenz(dt: float = 0.02, total_time: float = 300.0, initial=(1.0, 1.0, 1.0)):
    times = np.arange(0.0, total_time + 0.5 * dt, dt)
    solution = solve_ivp(
        lorenz_rhs,
        (times[0], times[-1]),
        np.asarray(initial, dtype=float),
        t_eval=times,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    states = solution.y.T
    derivatives = np.asarray([lorenz_rhs(t, x) for t, x in zip(times, states)])
    return times, states, derivatives


def simulate_mackey_glass(
    dt: float = 0.1,
    total_time: float = 5000.0,
    beta: float = 0.2,
    gamma: float = 0.1,
    exponent: int = 10,
    physical_delay: float = 17.0,
    history: float = 1.2,
):
    delay_steps = int(round(physical_delay / dt))
    total_steps = int(round(total_time / dt))
    padded = np.full(total_steps + delay_steps + 1, history, dtype=float)
    for index in range(delay_steps, delay_steps + total_steps):
        current = padded[index]
        delayed = padded[index - delay_steps]
        dx = beta * delayed / (1.0 + delayed**exponent) - gamma * current
        padded[index + 1] = current + dt * dx
    times = np.arange(total_steps + 1) * dt
    return times, padded[delay_steps : delay_steps + total_steps + 1]


def rfft_wavenumbers(n_nodes: int, domain_length: float):
    return 2.0 * np.pi * np.fft.rfftfreq(n_nodes, d=domain_length / n_nodes)


def dealiased_nonlinear_rfft(v: np.ndarray, wavenumbers: np.ndarray, n_nodes: int):
    n_padded = 3 * n_nodes // 2
    padded = np.zeros(n_padded // 2 + 1, dtype=complex)
    padded[: len(v)] = v * (n_padded / n_nodes)
    u_padded = np.fft.irfft(padded, n=n_padded)
    nonlinear_padded = np.fft.rfft(u_padded**2)
    return -0.5j * wavenumbers * nonlinear_padded[: len(v)] * (n_nodes / n_padded)


def ks_rhs(state: np.ndarray, domain_length: float = 22.0):
    n_nodes = len(state)
    k = rfft_wavenumbers(n_nodes, domain_length)
    state_hat = np.fft.rfft(state)
    rhs_hat = (k**2 - k**4) * state_hat + dealiased_nonlinear_rfft(state_hat, k, n_nodes)
    return np.fft.irfft(rhs_hat, n=n_nodes)


def ks_rhs_batch(states: np.ndarray, domain_length: float = 22.0):
    return np.asarray([ks_rhs(state, domain_length) for state in states])


def _ks_initial_condition(x: np.ndarray, domain_length: float, seed: int):
    rng = np.random.default_rng(seed)
    phase = 2.0 * np.pi * x / domain_length
    return np.cos(phase) * (1.0 + np.sin(phase)) + 0.05 * np.sin(3.0 * phase + 0.3) + 0.01 * rng.standard_normal(len(x))


def _simulate_ks_etdrk4(initial_state: np.ndarray, domain_length: float, dt: float, n_steps: int):
    n_nodes = len(initial_state)
    k = rfft_wavenumbers(n_nodes, domain_length)
    linear = k**2 - k**4
    E, E2 = np.exp(dt * linear), np.exp(0.5 * dt * linear)
    roots = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    LR = dt * linear[:, None] + roots[None, :]
    Q = dt * np.real(np.mean((np.exp(0.5 * LR) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))

    def nonlinear(v):
        return dealiased_nonlinear_rfft(v, k, n_nodes)

    state_hat = np.fft.rfft(initial_state)
    trajectory = np.zeros((n_steps + 1, n_nodes))
    trajectory[0] = initial_state
    for step in range(1, n_steps + 1):
        Nv = nonlinear(state_hat)
        a = E2 * state_hat + Q * Nv
        Na = nonlinear(a)
        b = E2 * state_hat + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlinear(c)
        state_hat = E * state_hat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        if n_nodes % 2 == 0:
            state_hat[-1] = state_hat[-1].real
        trajectory[step] = np.fft.irfft(state_hat, n=n_nodes)
    return trajectory


def simulate_ks(
    n_nodes: int = 14,
    output_dt: float = 0.1,
    internal_dt: float = 0.025,
    total_time: float = 370.0,
    domain_length: float = 22.0,
    seed: int = 1,
):
    x = np.linspace(0.0, domain_length, n_nodes, endpoint=False)
    save_every = int(round(output_dt / internal_dt))
    n_output_steps = int(round(total_time / output_dt))
    fine = _simulate_ks_etdrk4(
        _ks_initial_condition(x, domain_length, seed),
        domain_length,
        internal_dt,
        n_output_steps * save_every,
    )
    states = fine[::save_every]
    times = np.arange(len(states)) * output_dt
    return times, states, ks_rhs_batch(states, domain_length)

