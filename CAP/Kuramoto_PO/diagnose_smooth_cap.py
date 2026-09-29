"""Nonrigorous diagnostics for the smoothed infinite-dimensional KS proof.

This file is intentionally separate from the INTLAB certificate.  It builds
the finite block of the Figueras--de la Llave smoothed correction operator in
the real sine/cosine basis and is used to choose a proof truncation before an
expensive outward-rounded run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def real_time_coefficients(centered: np.ndarray, modes: int) -> np.ndarray:
    """Return [constant, cos(1:M), sin(1:M)] coefficients."""
    full_modes = (centered.shape[0] - 1) // 2
    if modes > full_modes:
        raise ValueError("requested more temporal modes than the candidate")
    out = np.empty((2 * modes + 1, centered.shape[1]), dtype=float)
    out[0] = centered[full_modes].real
    positive = centered[full_modes + 1 : full_modes + modes + 1]
    out[1 : modes + 1] = 2 * positive.real
    out[modes + 1 :] = -2 * positive.imag
    return out


def temporal_evaluation(modes: int, phases: np.ndarray) -> np.ndarray:
    out = np.empty((len(phases), 2 * modes + 1))
    out[:, 0] = 1.0
    harmonics = np.arange(1, modes + 1)
    out[:, 1 : modes + 1] = np.cos(phases[:, None] * harmonics)
    out[:, modes + 1 :] = np.sin(phases[:, None] * harmonics)
    return out


def temporal_projection(modes: int, phases: np.ndarray) -> np.ndarray:
    evaluation = temporal_evaluation(modes, phases)
    projection = (2.0 / len(phases)) * evaluation.T
    projection[0] *= 0.5
    return projection


def temporal_derivative(modes: int) -> np.ndarray:
    out = np.zeros((2 * modes + 1, 2 * modes + 1))
    for harmonic in range(1, modes + 1):
        out[modes + harmonic, harmonic] = -harmonic
        out[harmonic, modes + harmonic] = harmonic
    return out


def spatial_product_derivative_matrix(u: np.ndarray, modes: int) -> np.ndarray:
    """Matrix of delta -> d_x(u*delta), projected to modes 1:modes."""
    out = np.zeros((modes, modes))
    for input_mode in range(1, modes + 1):
        for u_mode, coefficient in enumerate(u, start=1):
            difference = abs(u_mode - input_mode)
            if 1 <= difference <= modes:
                out[difference - 1, input_mode - 1] -= 0.5 * difference * coefficient
            total = u_mode + input_mode
            if total <= modes:
                out[total - 1, input_mode - 1] += 0.5 * total * coefficient
    return out


def temporal_product_matrix(
    centered_u: np.ndarray, output_modes: int, input_modes: int
) -> np.ndarray:
    """Real trigonometric matrix for multiplication by a real Fourier series."""
    centered_u = np.asarray(centered_u, complex)
    u_modes = (len(centered_u) - 1) // 2

    def coefficient(index: np.ndarray) -> np.ndarray:
        values = np.zeros(index.shape, complex)
        valid = np.abs(index) <= u_modes
        values[valid] = centered_u[index[valid] + u_modes]
        return values

    output_indices = np.arange(-output_modes, output_modes + 1)
    complex_columns = []
    complex_columns.append(coefficient(output_indices))
    for harmonic in range(1, input_modes + 1):
        complex_columns.append(
            0.5
            * (
                coefficient(output_indices - harmonic)
                + coefficient(output_indices + harmonic)
            )
        )
    for harmonic in range(1, input_modes + 1):
        complex_columns.append(
            0.5j
            * (
                coefficient(output_indices + harmonic)
                - coefficient(output_indices - harmonic)
            )
        )
    complex_matrix = np.column_stack(complex_columns)
    real_matrix = np.empty((2 * output_modes + 1, 2 * input_modes + 1))
    real_matrix[0] = complex_matrix[output_modes].real
    positive = complex_matrix[output_modes + 1 :]
    real_matrix[1 : output_modes + 1] = 2 * positive.real
    real_matrix[output_modes + 1 :] = -2 * positive.imag
    return real_matrix


def core_multiplication_rows(
    centered: np.ndarray,
    output_temporal_modes: int,
    output_spatial_modes: int,
    input_temporal_modes: int,
    input_spatial_modes: int,
) -> np.ndarray:
    """Rows in the core rectangle of d_x(u*delta) for a larger input block."""
    u_temporal_modes = (centered.shape[0] - 1) // 2
    u_spatial_modes = centered.shape[1]
    rows = (2 * output_temporal_modes + 1) * output_spatial_modes
    columns = (2 * input_temporal_modes + 1) * input_spatial_modes
    out = np.zeros((rows, columns))
    for output_space in range(1, output_spatial_modes + 1):
        for input_space in range(1, input_spatial_modes + 1):
            temporal_series = np.zeros(2 * u_temporal_modes + 1, complex)
            # sin(ku*x)sin(l*x), followed by d/dx and projection to n.
            sum_mode = output_space - input_space
            if 1 <= sum_mode <= u_spatial_modes:
                temporal_series += 0.5 * output_space * centered[:, sum_mode - 1]
            for difference_mode in (input_space - output_space, input_space + output_space):
                if 1 <= difference_mode <= u_spatial_modes:
                    temporal_series -= (
                        0.5 * output_space * centered[:, difference_mode - 1]
                    )
            if not np.any(temporal_series):
                continue
            temporal_block = temporal_product_matrix(
                temporal_series, output_temporal_modes, input_temporal_modes
            )
            output_indices = (
                np.arange(2 * output_temporal_modes + 1) * output_spatial_modes
                + output_space
                - 1
            )
            input_indices = (
                np.arange(2 * input_temporal_modes + 1) * input_spatial_modes
                + input_space
                - 1
            )
            out[np.ix_(output_indices, input_indices)] = temporal_block
    return out


def apply_sinv_real(
    coefficients: np.ndarray, frequency: float, nu: float
) -> np.ndarray:
    """Apply the diagonal S_c inverse to real time/space coefficients."""
    coefficients = np.asarray(coefficients, float)
    temporal_modes = (coefficients.shape[0] - 1) // 2
    c = 1.0 / nu
    spatial = np.arange(1, coefficients.shape[1] + 1, dtype=float)
    harmonic = np.arange(1, temporal_modes + 1, dtype=float)[:, None]
    p = nu * spatial**4 - spatial**2 + c
    temporal = frequency * harmonic
    denominator = p[None, :] ** 2 + temporal**2
    cosine = coefficients[1 : temporal_modes + 1]
    sine = coefficients[temporal_modes + 1 :]
    out = np.empty_like(coefficients)
    out[0] = coefficients[0] / p
    out[1 : temporal_modes + 1] = (
        p[None, :] * cosine - temporal * sine
    ) / denominator
    out[temporal_modes + 1 :] = (
        temporal * cosine + p[None, :] * sine
    ) / denominator
    return out


def apply_raw_linear_inverse_real(
    coefficients: np.ndarray, frequency: float, nu: float
) -> np.ndarray:
    """Apply (f*d_theta+nu*d_x^4+d_x^2)^-1 mode by mode."""
    coefficients = np.asarray(coefficients, float)
    temporal_modes = (coefficients.shape[0] - 1) // 2
    spatial = np.arange(1, coefficients.shape[1] + 1, dtype=float)
    harmonic = np.arange(1, temporal_modes + 1, dtype=float)[:, None]
    p = nu * spatial**4 - spatial**2
    temporal = frequency * harmonic
    denominator = p[None, :] ** 2 + temporal**2
    cosine = coefficients[1 : temporal_modes + 1]
    sine = coefficients[temporal_modes + 1 :]
    out = np.empty_like(coefficients)
    out[0] = coefficients[0] / p
    out[1 : temporal_modes + 1] = (
        p[None, :] * cosine - temporal * sine
    ) / denominator
    out[temporal_modes + 1 :] = (
        temporal * cosine + p[None, :] * sine
    ) / denominator
    return out


def direct_product_derivative_column(
    centered: np.ndarray,
    input_harmonic: int,
    input_is_sine: bool,
    input_spatial_mode: int,
) -> np.ndarray:
    """Return ``d_x(u*phi)`` without constructing a temporal product matrix.

    ``centered`` contains complex temporal Fourier coefficients of the real
    sine amplitudes of ``u``.  The input basis function is either
    ``cos(input_harmonic*theta) sin(input_spatial_mode*x)`` or its sine
    counterpart.  For harmonic zero only the cosine (constant) input is
    meaningful.  This direct shift/convolution form is fast enough to scan
    every finite/tail interface column in the proof diagnostics.
    """
    centered = np.asarray(centered, complex)
    u_temporal_modes = (centered.shape[0] - 1) // 2
    output_temporal_modes = u_temporal_modes + input_harmonic
    temporal_product = np.zeros(
        (2 * output_temporal_modes + 1, centered.shape[1]), complex
    )
    offset = output_temporal_modes - u_temporal_modes
    if input_harmonic == 0:
        if input_is_sine:
            raise ValueError("the zero temporal harmonic has no sine basis vector")
        temporal_product[offset : offset + len(centered)] = centered
    else:
        rows = np.arange(len(centered)) + offset
        if input_is_sine:
            temporal_product[rows + input_harmonic] += -0.5j * centered
            temporal_product[rows - input_harmonic] += 0.5j * centered
        else:
            temporal_product[rows + input_harmonic] += 0.5 * centered
            temporal_product[rows - input_harmonic] += 0.5 * centered

    output_spatial_modes = input_spatial_mode + centered.shape[1]
    product = np.zeros(
        (2 * output_temporal_modes + 1, output_spatial_modes), complex
    )
    for u_mode in range(1, centered.shape[1] + 1):
        difference = abs(u_mode - input_spatial_mode)
        if difference:
            product[:, difference - 1] -= (
                0.5 * difference * temporal_product[:, u_mode - 1]
            )
        total = u_mode + input_spatial_mode
        product[:, total - 1] += 0.5 * total * temporal_product[:, u_mode - 1]

    real_product = np.empty_like(product.real)
    real_product[0] = product[output_temporal_modes].real
    positive = product[output_temporal_modes + 1 :]
    real_product[1 : output_temporal_modes + 1] = 2.0 * positive.real
    real_product[output_temporal_modes + 1 :] = -2.0 * positive.imag
    return real_product


def direct_ahat_column(
    centered: np.ndarray,
    input_harmonic: int,
    input_is_sine: bool,
    input_spatial_mode: int,
    frequency: float,
    nu: float,
) -> np.ndarray:
    """Direct column of ``-c*S_c^-1 + S_c^-1*d_x(u*.)``."""
    product = direct_product_derivative_column(
        centered, input_harmonic, input_is_sine, input_spatial_mode
    )
    result = apply_sinv_real(product, frequency, nu)
    temporal_modes = (len(result) - 1) // 2
    row = (
        temporal_modes + input_harmonic
        if input_is_sine
        else input_harmonic
    )
    unit = np.zeros_like(result)
    unit[row, input_spatial_mode - 1] = 1.0
    result -= (1.0 / nu) * apply_sinv_real(unit, frequency, nu)
    return result


def direct_raw_tail_column(
    centered: np.ndarray,
    input_harmonic: int,
    input_is_sine: bool,
    input_spatial_mode: int,
    frequency: float,
    nu: float,
) -> np.ndarray:
    """Direct column of ``L_0^-1*d_x(u*.)`` for the diagonal tail inverse."""
    product = direct_product_derivative_column(
        centered, input_harmonic, input_is_sine, input_spatial_mode
    )
    return apply_raw_linear_inverse_real(product, frequency, nu)


def build_symmetry_finite_operator_direct(
    candidate: dict, temporal_modes: int, spatial_modes: int
):
    """Build the n+k-even finite correction operator by Fourier shifts.

    Unlike :func:`build_finite_operator`, this routine contains no
    trigonometric evaluation or quadrature.  Each multiplication entry is a
    sum of at most six stored Fourier coefficients, which makes a conventional
    floating-point forward-error audit directly applicable.
    """
    ell = float(np.squeeze(candidate["interval_length"]))
    q = np.pi / ell
    nu = q * q
    frequency = float(np.squeeze(candidate["frequency"])) / nu
    centered = np.asarray(candidate["velocity_time_fourier"], complex) / q
    candidate_modes = (len(centered) - 1) // 2
    for row, harmonic in enumerate(range(-candidate_modes, candidate_modes + 1)):
        for spatial_mode in range(1, centered.shape[1] + 1):
            if (harmonic + spatial_mode) % 2:
                centered[row, spatial_mode - 1] = 0.0

    state_dimension = (2 * temporal_modes + 1) * spatial_modes
    derivative = np.kron(
        temporal_derivative(temporal_modes), np.eye(spatial_modes)
    )
    c = 1.0 / nu
    sinv = np.zeros((state_dimension, state_dimension))
    for spatial_index in range(spatial_modes):
        k = spatial_index + 1
        p = nu * k**4 - k**2 + c
        sinv[spatial_index, spatial_index] = 1.0 / p
        for harmonic in range(1, temporal_modes + 1):
            cosine = harmonic * spatial_modes + spatial_index
            sine = (temporal_modes + harmonic) * spatial_modes + spatial_index
            denominator = p * p + (frequency * harmonic) ** 2
            sinv[np.ix_([cosine, sine], [cosine, sine])] = np.asarray(
                [[p, -frequency * harmonic], [frequency * harmonic, p]]
            ) / denominator

    harmonics = np.r_[
        0,
        np.arange(1, temporal_modes + 1),
        np.arange(1, temporal_modes + 1),
    ]
    allowed = np.flatnonzero(
        (
            np.repeat(harmonics, spatial_modes)
            + np.tile(
                np.arange(1, spatial_modes + 1), 2 * temporal_modes + 1
            )
        )
        % 2
        == 0
    )
    sinv_allowed = sinv[np.ix_(allowed, allowed)]
    derivative_allowed = derivative[np.ix_(allowed, allowed)]
    candidate_real = real_time_coefficients(
        centered[:, :spatial_modes], temporal_modes
    ).reshape(-1)[allowed]
    tangent = derivative_allowed @ candidate_real
    selected_harmonics = np.repeat(harmonics, spatial_modes)[allowed]
    phase_weights = np.where(selected_harmonics == 0, 2.0, 1.0)
    phase_row = phase_weights * tangent
    frequency_column = sinv_allowed @ tangent

    multiplication = np.zeros((len(allowed), len(allowed)))
    for column_index, full_index in enumerate(allowed):
        time_basis, spatial_index = divmod(int(full_index), spatial_modes)
        if time_basis == 0:
            harmonic = 0
            is_sine = False
        elif time_basis <= temporal_modes:
            harmonic = time_basis
            is_sine = False
        else:
            harmonic = time_basis - temporal_modes
            is_sine = True
        product = direct_product_derivative_column(
            centered, harmonic, is_sine, spatial_index + 1
        )
        full_modes = (len(product) - 1) // 2
        rows = [0]
        rows.extend(range(1, temporal_modes + 1))
        rows.extend(
            range(full_modes + 1, full_modes + temporal_modes + 1)
        )
        multiplication[:, column_index] = product[
            np.ix_(rows, range(spatial_modes))
        ].reshape(-1)[allowed]

    state = (
        np.eye(len(allowed))
        - c * sinv_allowed
        + sinv_allowed @ multiplication
    )
    operator = np.block(
        [
            [np.zeros((1, 1)), phase_row[None, :]],
            [frequency_column[:, None], state],
        ]
    )
    return operator, allowed, {
        "nu": nu,
        "c": c,
        "frequency": frequency,
        "sinv_allowed": sinv_allowed,
        "derivative_allowed": derivative_allowed,
        "harmonics": harmonics,
    }


def ahat_column(
    centered: np.ndarray,
    input_temporal_modes: int,
    input_time_basis: int,
    input_spatial_mode: int,
    frequency: float,
    nu: float,
) -> np.ndarray:
    """One full column of -cS^-1+S^-1*d_x(u*.) in real coefficients."""
    u_temporal_modes = (centered.shape[0] - 1) // 2
    output_temporal_modes = input_temporal_modes + u_temporal_modes
    output_spatial_modes = input_spatial_mode + centered.shape[1]
    product = np.zeros((2 * output_temporal_modes + 1, output_spatial_modes))
    for output_space in range(1, output_spatial_modes + 1):
        temporal_series = np.zeros(2 * u_temporal_modes + 1, complex)
        sum_mode = output_space - input_spatial_mode
        if 1 <= sum_mode <= centered.shape[1]:
            temporal_series += 0.5 * output_space * centered[:, sum_mode - 1]
        for difference_mode in (
            input_spatial_mode - output_space,
            input_spatial_mode + output_space,
        ):
            if 1 <= difference_mode <= centered.shape[1]:
                temporal_series -= 0.5 * output_space * centered[:, difference_mode - 1]
        if np.any(temporal_series):
            block = temporal_product_matrix(
                temporal_series, output_temporal_modes, input_temporal_modes
            )
            product[:, output_space - 1] = block[:, input_time_basis]
    result = apply_sinv_real(product, frequency, nu)
    # The input basis embeds at a shifted sine-basis index when output and
    # input temporal truncations differ.
    if input_time_basis == 0:
        embedded_time = 0
    elif input_time_basis <= input_temporal_modes:
        embedded_time = input_time_basis
    else:
        harmonic = input_time_basis - input_temporal_modes
        embedded_time = output_temporal_modes + harmonic
    unit = np.zeros_like(product)
    unit[embedded_time, input_spatial_mode - 1] = 1.0
    result -= (1.0 / nu) * apply_sinv_real(unit, frequency, nu)
    return result


def tail_preconditioned_nonlinear_column(
    centered: np.ndarray,
    input_temporal_modes: int,
    input_time_basis: int,
    input_spatial_mode: int,
    frequency: float,
    nu: float,
) -> np.ndarray:
    """Full column L^-1*d_x(u*.) used by the diagonal tail inverse."""
    u_temporal_modes = (centered.shape[0] - 1) // 2
    output_temporal_modes = input_temporal_modes + u_temporal_modes
    output_spatial_modes = input_spatial_mode + centered.shape[1]
    product = np.zeros((2 * output_temporal_modes + 1, output_spatial_modes))
    for output_space in range(1, output_spatial_modes + 1):
        temporal_series = np.zeros(2 * u_temporal_modes + 1, complex)
        sum_mode = output_space - input_spatial_mode
        if 1 <= sum_mode <= centered.shape[1]:
            temporal_series += 0.5 * output_space * centered[:, sum_mode - 1]
        for difference_mode in (
            input_spatial_mode - output_space,
            input_spatial_mode + output_space,
        ):
            if 1 <= difference_mode <= centered.shape[1]:
                temporal_series -= 0.5 * output_space * centered[:, difference_mode - 1]
        if np.any(temporal_series):
            block = temporal_product_matrix(
                temporal_series, output_temporal_modes, input_temporal_modes
            )
            product[:, output_space - 1] = block[:, input_time_basis]
    return apply_raw_linear_inverse_real(product, frequency, nu)


def build_finite_operator(candidate: dict, temporal_modes: int, spatial_modes: int):
    ell = float(np.squeeze(candidate["interval_length"]))
    q = np.pi / ell
    nu = q * q
    frequency = float(np.squeeze(candidate["frequency"])) / nu
    centered = np.asarray(candidate["velocity_time_fourier"], complex) / q
    full_temporal_modes = (centered.shape[0] - 1) // 2
    # Only u modes up to twice the finite rectangle can contribute to its
    # projected multiplication block.
    u_t_modes = min(2 * temporal_modes, full_temporal_modes)
    u_x_modes = min(2 * spatial_modes, centered.shape[1])
    u_coeff = real_time_coefficients(centered[:, :u_x_modes], u_t_modes)

    quadrature_count = 2 * (u_t_modes + temporal_modes) + 1
    phases = 2 * np.pi * np.arange(quadrature_count) / quadrature_count
    eval_u = temporal_evaluation(u_t_modes, phases)
    eval_in = temporal_evaluation(temporal_modes, phases)
    project_out = temporal_projection(temporal_modes, phases)
    u_values = eval_u @ u_coeff
    spatial_blocks = np.asarray(
        [spatial_product_derivative_matrix(row, spatial_modes) for row in u_values]
    )
    # Axes: output time, output space, input time, input space.
    multiplication = np.einsum(
        "an,nkl,nb->akbl", project_out, spatial_blocks, eval_in, optimize=True
    ).reshape(((2 * temporal_modes + 1) * spatial_modes,) * 2)

    dt = temporal_derivative(temporal_modes)
    state_dimension = (2 * temporal_modes + 1) * spatial_modes
    derivative = np.kron(dt, np.eye(spatial_modes))
    c = 1.0 / nu
    sinv = np.zeros((state_dimension, state_dimension))
    for spatial_index in range(spatial_modes):
        k = spatial_index + 1
        p = nu * k**4 - k**2 + c
        base = spatial_index
        sinv[base, base] = 1.0 / p
        for harmonic in range(1, temporal_modes + 1):
            cosine = harmonic * spatial_modes + spatial_index
            sine = (temporal_modes + harmonic) * spatial_modes + spatial_index
            denominator = p * p + (frequency * harmonic) ** 2
            sinv[np.ix_([cosine, sine], [cosine, sine])] = np.asarray(
                [[p, -frequency * harmonic], [frequency * harmonic, p]]
            ) / denominator

    candidate_finite = real_time_coefficients(
        centered[:, :spatial_modes], temporal_modes
    ).reshape(-1)
    tangent = derivative @ candidate_finite
    time_weights = np.r_[2.0, np.ones(2 * temporal_modes)]
    l2_weights = np.repeat(time_weights, spatial_modes)
    phase_row = l2_weights * tangent
    phase_scale = float(np.dot(phase_row, tangent))

    state = np.eye(state_dimension) - c * sinv + sinv @ multiplication
    frequency_column = sinv @ tangent
    operator = np.block(
        [[np.zeros((1, 1)), phase_row[None, :]], [frequency_column[:, None], state]]
    )
    return operator, {
        "nu": nu,
        "c": c,
        "frequency": frequency,
        "multiplication_norm": np.linalg.norm(multiplication, 1),
        "phase_scale": phase_scale,
        "sinv": sinv,
        "time_derivative": derivative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-modes", type=int, default=24)
    parser.add_argument("--spatial-modes", type=int, default=16)
    parser.add_argument("--cross-diagnostic", action="store_true")
    parser.add_argument(
        "--cross-buffer",
        type=int,
        default=None,
        help="limit the necessary cross diagnostic to this boundary width",
    )
    parser.add_argument(
        "--weight-r",
        type=float,
        default=0.0,
        help="common exponential Fourier weight exp(r*(time+space))",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "ks_neumann_cap_candidate.mat",
    )
    args = parser.parse_args()
    operator, constants = build_finite_operator(
        loadmat(args.candidate), args.temporal_modes, args.spatial_modes
    )
    inverse = np.linalg.inv(operator)
    defect = np.eye(len(operator)) - inverse @ operator
    print(f"dimension={len(operator)}")
    for key, value in constants.items():
        if np.isscalar(value):
            print(f"{key}={value:.16e}")
    state_inverse = inverse[:, 1:]
    theta_smoothed = state_inverse @ constants["sinv"] @ constants["time_derivative"]
    print(f"B_Sinv_Dtheta_norm_1={np.linalg.norm(theta_smoothed, 1):.16e}")
    print(f"condition_1={np.linalg.cond(operator, 1):.16e}")
    print(f"inverse_norm_1={np.linalg.norm(inverse, 1):.16e}")
    print(f"floating_defect_norm_1={np.linalg.norm(defect, 1):.16e}")
    if args.cross_diagnostic:
        ell = float(np.squeeze(loadmat(args.candidate)["interval_length"]))
        q = np.pi / ell
        centered = (
            np.asarray(loadmat(args.candidate)["velocity_time_fourier"], complex)
            / q
        )
        if args.cross_buffer is None:
            extended_t = args.temporal_modes + (centered.shape[0] - 1) // 2
            extended_x = args.spatial_modes + centered.shape[1]
        else:
            extended_t = args.temporal_modes + args.cross_buffer
            extended_x = args.spatial_modes + args.cross_buffer
        core_rows = core_multiplication_rows(
            centered,
            args.temporal_modes,
            args.spatial_modes,
            extended_t,
            extended_x,
        )
        # Apply S_c^{-1} on the core output and include -cS_c^{-1} on
        # columns belonging to the core.  Only tail-input columns matter for
        # this necessary cross-block diagnostic.
        core_ahat = constants["sinv"] @ core_rows
        core_time_indices = [0]
        core_time_indices.extend(range(1, args.temporal_modes + 1))
        core_time_indices.extend(
            range(
                extended_t + 1,
                extended_t + args.temporal_modes + 1,
            )
        )
        core_input_indices = []
        for time_index in core_time_indices:
            start = time_index * extended_x
            core_input_indices.extend(range(start, start + args.spatial_modes))
        tail_mask = np.ones(core_ahat.shape[1], dtype=bool)
        tail_mask[core_input_indices] = False
        core_harmonics = np.r_[
            0,
            np.arange(1, args.temporal_modes + 1),
            np.arange(1, args.temporal_modes + 1),
        ]
        core_state_weights = np.exp(
            args.weight_r
            * (
                np.repeat(core_harmonics, args.spatial_modes)
                + np.tile(np.arange(1, args.spatial_modes + 1), len(core_harmonics))
            )
        )
        product_output_weights = np.r_[1.0, core_state_weights]
        extended_harmonics = np.r_[
            0,
            np.arange(1, extended_t + 1),
            np.arange(1, extended_t + 1),
        ]
        extended_input_weights = np.exp(
            args.weight_r
            * (
                np.repeat(extended_harmonics, extended_x)
                + np.tile(np.arange(1, extended_x + 1), len(extended_harmonics))
            )
        )
        maximum = 0.0
        maximizing_column = -1
        state_inverse = inverse[:, 1:]
        tail_columns = np.flatnonzero(tail_mask)
        for start in range(0, len(tail_columns), 256):
            selected = tail_columns[start : start + 256]
            transformed = state_inverse @ core_ahat[:, selected]
            transformed *= product_output_weights[:, None]
            transformed /= extended_input_weights[selected][None, :]
            column_norms = np.sum(np.abs(transformed), axis=0)
            local = int(np.argmax(column_norms))
            if float(column_norms[local]) > maximum:
                maximum = float(column_norms[local])
                maximizing_column = int(selected[local])
        print(f"necessary_BA_finite_tail_cross_norm_1={maximum:.16e}")
        print(
            "cross_maximizing_input_time_basis="
            f"{maximizing_column // extended_x},"
            "input_spatial_mode="
            f"{maximizing_column % extended_x + 1}"
        )
        input_time_basis = maximizing_column // extended_x
        input_space = maximizing_column % extended_x + 1
        full_column = ahat_column(
            centered,
            extended_t,
            input_time_basis,
            input_space,
            constants["frequency"],
            constants["nu"],
        )
        full_modes = (full_column.shape[0] - 1) // 2
        core_time_output = [0]
        core_time_output.extend(range(1, args.temporal_modes + 1))
        core_time_output.extend(
            range(full_modes + 1, full_modes + args.temporal_modes + 1)
        )
        core_output_mask = np.zeros(full_column.shape, dtype=bool)
        core_output_mask[core_time_output, : args.spatial_modes] = True
        full_harmonics = np.r_[
            0, np.arange(1, full_modes + 1), np.arange(1, full_modes + 1)
        ]
        full_weights = np.exp(
            args.weight_r
            * (
                full_harmonics[:, None]
                + np.arange(1, full_column.shape[1] + 1)[None, :]
            )
        )
        maximizing_input_weight = extended_input_weights[maximizing_column]
        tail_part = float(
            np.sum(
                np.abs(full_column[~core_output_mask])
                * full_weights[~core_output_mask]
            )
            / maximizing_input_weight
        )
        print(f"same_column_tail_output_norm_1={tail_part:.16e}")
        print(f"same_column_full_defect_lower_bound={maximum + tail_part:.16e}")
        tail_preconditioned = tail_preconditioned_nonlinear_column(
            centered,
            extended_t,
            input_time_basis,
            input_space,
            constants["frequency"],
            constants["nu"],
        )
        tail_modes = (tail_preconditioned.shape[0] - 1) // 2
        tail_core_times = [0]
        tail_core_times.extend(range(1, args.temporal_modes + 1))
        tail_core_times.extend(
            range(tail_modes + 1, tail_modes + args.temporal_modes + 1)
        )
        tail_mask_output = np.ones(tail_preconditioned.shape, dtype=bool)
        tail_mask_output[tail_core_times, : args.spatial_modes] = False
        tail_harmonics = np.r_[
            0, np.arange(1, tail_modes + 1), np.arange(1, tail_modes + 1)
        ]
        tail_weights = np.exp(
            args.weight_r
            * (
                tail_harmonics[:, None]
                + np.arange(1, tail_preconditioned.shape[1] + 1)[None, :]
            )
        )
        diagonal_tail_part = float(
            np.sum(
                np.abs(tail_preconditioned[tail_mask_output])
                * tail_weights[tail_mask_output]
            )
            / maximizing_input_weight
        )
        print(
            "same_column_diagonal_tail_preconditioned_norm_1="
            f"{diagonal_tail_part:.16e}"
        )
        print(
            "same_column_block_diagonal_defect_lower_bound="
            f"{maximum + diagonal_tail_part:.16e}"
        )


if __name__ == "__main__":
    main()
