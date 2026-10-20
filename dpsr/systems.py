"""Reference solvers and observation coordinates for MG and periodic KS.

These generate numerical reference data, not rigorous PDE or DDE enclosures.
KS configuration is passed explicitly; its nodal resolution defines the
semidiscrete reference system.
"""
import numpy as np


def generate_mackey_glass_euler(
    beta=0.2,
    gamma=0.1,
    exponent=10,
    delay=17.0,
    dt=0.1,
    t_max=5000.0,
    history_value=1.2,
):
    """Generate x(t) with explicit Euler and a constant prehistory."""
    delay_steps = int(round(delay / dt))
    total_steps = int(round(t_max / dt))
    values = np.full(total_steps + delay_steps + 1, history_value, dtype=float)

    for i in range(delay_steps, delay_steps + total_steps):
        x_now = values[i]
        x_delay = values[i - delay_steps]
        derivative = beta * x_delay / (1.0 + x_delay**exponent) - gamma * x_now
        values[i + 1] = x_now + dt * derivative

    time = np.arange(total_steps + 1) * dt
    series = values[delay_steps : delay_steps + total_steps + 1]
    return time, series


def build_delay_embedding(series, start_step, end_step, delay_steps, dimension):
    """Build aligned columns [x(t), x(t-delay), ..., x(t-(d-1)delay)]."""
    earliest = start_step - (dimension - 1) * delay_steps
    if earliest < 0:
        raise ValueError("The requested embedding reaches before the available series.")
    if end_step > len(series):
        raise ValueError("The requested embedding extends beyond the available series.")
    columns = [
        series[start_step - j * delay_steps : end_step - j * delay_steps]
        for j in range(dimension)
    ]
    return np.column_stack(columns)



def ks_initial_condition(x, domain_length, seed=1):
    """Deterministic large-scale structure plus a small seeded perturbation."""
    local_rng = np.random.default_rng(seed)
    phase = 2.0 * np.pi * x / domain_length
    return (
        np.cos(phase) * (1.0 + np.sin(phase))
        + 0.05 * np.sin(3.0 * phase + 0.3)
        + 0.01 * local_rng.standard_normal(len(x))
    )


def rfft_wavenumbers(n_nodes, domain_length):
    return 2.0 * np.pi * np.fft.rfftfreq(
        n_nodes, d=domain_length / n_nodes
    )


def dealiased_nonlinear_rfft(v, wavenumbers, n_nodes):
    """Return the Fourier coefficients of -u*u_x using 3/2 padding."""
    n_padded = 3 * n_nodes // 2
    padded = np.zeros(n_padded // 2 + 1, dtype=complex)
    padded[: len(v)] = v * (n_padded / n_nodes)
    u_padded = np.fft.irfft(padded, n=n_padded)
    nonlinear_padded = np.fft.rfft(u_padded**2)
    nonlinear = nonlinear_padded[: len(v)] * (n_nodes / n_padded)
    return -0.5j * wavenumbers * nonlinear


def ks_rhs_spectral_state(state, domain_length):
    """Evaluate the semidiscrete KS right-hand side at one nodal state."""
    n_nodes = len(state)
    wavenumbers = rfft_wavenumbers(n_nodes, domain_length)
    state_hat = np.fft.rfft(state)
    linear_operator = wavenumbers**2 - wavenumbers**4
    rhs_hat = (
        linear_operator * state_hat
        + dealiased_nonlinear_rfft(state_hat, wavenumbers, n_nodes)
    )
    return np.fft.irfft(rhs_hat, n=n_nodes)


def ks_rhs_spectral_batch(states, domain_length):
    """Evaluate the exact spectral derivative for each row of a trajectory."""
    return np.asarray(
        [ks_rhs_spectral_state(state, domain_length) for state in states]
    )


def simulate_ks_etdrk4(initial_state, domain_length, dt, n_steps):
    """Integrate the semidiscrete KS system with ETDRK4."""
    n_nodes = len(initial_state)
    k = rfft_wavenumbers(n_nodes, domain_length)
    linear = k**2 - k**4
    E = np.exp(dt * linear)
    E2 = np.exp(0.5 * dt * linear)

    quadrature_nodes = 32
    roots = np.exp(
        1j
        * np.pi
        * (np.arange(1, quadrature_nodes + 1) - 0.5)
        / quadrature_nodes
    )
    LR = dt * linear[:, None] + roots[None, :]
    Q = dt * np.real(np.mean((np.exp(0.5 * LR) - 1.0) / LR, axis=1))
    f1 = dt * np.real(
        np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2))
            / LR**3,
            axis=1,
        )
    )
    f2 = dt * np.real(
        np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1)
    )
    f3 = dt * np.real(
        np.mean(
            (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR))
            / LR**3,
            axis=1,
        )
    )

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


def simulate_ks_nodes(n_nodes, config, seed=None):
    """Simulate KS with internal substeps and return x, output times, and states."""
    seed = config.random_seed if seed is None else seed
    ratio = config.output_dt / config.internal_dt
    save_every = int(round(ratio))
    if not np.isclose(save_every * config.internal_dt, config.output_dt):
        raise ValueError("output_dt must be an integer multiple of internal_dt.")

    x = np.linspace(0.0, config.domain_length, n_nodes, endpoint=False)
    times = np.arange(
        0.0, config.total_time + 0.5 * config.output_dt, config.output_dt
    )
    fine = simulate_ks_etdrk4(
        ks_initial_condition(x, config.domain_length, seed),
        config.domain_length,
        config.internal_dt,
        (len(times) - 1) * save_every,
    )
    states = fine[::save_every]

    if not np.all(np.isfinite(states)):
        raise RuntimeError("KS solver produced non-finite values.")
    if np.max(np.abs(states)) > 50.0:
        raise RuntimeError("KS solver exceeded the configured sanity bound.")
    return x, times, states
