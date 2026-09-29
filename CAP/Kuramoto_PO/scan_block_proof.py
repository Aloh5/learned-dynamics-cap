"""Nonrigorous full-interface scan for the symmetry-reduced KS CAP.

The script is a dimension-selection diagnostic, not the final certificate.
It evaluates every finite input column and every tail input column that can
couple back into the finite rectangle.  The final MATLAB/INTLAB driver repeats
the selected estimates with outward rounding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from diagnose_smooth_cap import (
    build_symmetry_finite_operator_direct,
    direct_ahat_column,
    direct_raw_tail_column,
)


def project_half_shift_symmetry(centered: np.ndarray) -> np.ndarray:
    """Project onto u(theta+pi,x)=-u(theta,pi-x), i.e. n+k even."""
    centered = np.asarray(centered, complex).copy()
    modes = (len(centered) - 1) // 2
    for row, harmonic in enumerate(range(-modes, modes + 1)):
        for spatial_mode in range(1, centered.shape[1] + 1):
            if (harmonic + spatial_mode) % 2:
                centered[row, spatial_mode - 1] = 0.0
    # Canonical real polynomial: avoid treating FFT conjugacy roundoff as
    # different coefficients in the numerical and interval operators.
    centered[modes] = centered[modes].real
    centered[:modes] = centered[modes+1:][::-1].conj()
    return centered


def core_layout(temporal_modes: int, spatial_modes: int):
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
    return harmonics, allowed


def core_part(column, temporal_modes, spatial_modes, allowed):
    full_modes = (len(column) - 1) // 2
    rows = [0]
    rows.extend(range(1, temporal_modes + 1))
    rows.extend(range(full_modes + 1, full_modes + temporal_modes + 1))
    return column[np.ix_(rows, range(spatial_modes))].reshape(-1)[allowed]


def tail_norm(column, temporal_modes, spatial_modes):
    full_modes = (len(column) - 1) // 2
    rows = [0]
    rows.extend(range(1, temporal_modes + 1))
    rows.extend(range(full_modes + 1, full_modes + temporal_modes + 1))
    mask = np.ones(column.shape, dtype=bool)
    mask[np.ix_(rows, range(spatial_modes))] = False
    return float(np.sum(np.abs(column[mask])))


def basis_vectors(max_harmonic: int, max_spatial_mode: int):
    for harmonic in range(max_harmonic + 1):
        for spatial_mode in range(1, max_spatial_mode + 1):
            if (harmonic + spatial_mode) % 2:
                continue
            yield harmonic, False, spatial_mode
            if harmonic:
                yield harmonic, True, spatial_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-modes", type=int, default=160)
    parser.add_argument("--spatial-modes", type=int, default=40)
    parser.add_argument("--refine-threshold", type=float, default=0.88)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "ks_neumann_cap_candidate.mat",
    )
    args = parser.parse_args()
    data = loadmat(args.candidate)
    ell = float(np.squeeze(data["interval_length"]))
    q = np.pi / ell
    nu = q * q
    frequency = float(np.squeeze(data["frequency"])) / nu
    physical = project_half_shift_symmetry(np.asarray(data["velocity_time_fourier"], complex))
    centered = physical / q
    candidate_t = (len(centered) - 1) // 2
    candidate_x = centered.shape[1]

    projected_data = dict(data)
    projected_data["velocity_time_fourier"] = physical
    finite_operator, allowed, constants = build_symmetry_finite_operator_direct(
        projected_data, args.temporal_modes, args.spatial_modes
    )
    inverse = np.linalg.inv(finite_operator)
    finite_defect = float(
        np.linalg.norm(np.eye(len(inverse)) - inverse @ finite_operator, 1)
    )
    inverse_state = inverse[:, 1:]
    inverse_column_norms = np.sum(np.abs(inverse_state), axis=0)

    maximum_finite_tail = (0.0, None)
    for harmonic, is_sine, spatial_mode in basis_vectors(
        args.temporal_modes, args.spatial_modes
    ):
        raw = direct_raw_tail_column(
            centered,
            harmonic,
            is_sine,
            spatial_mode,
            frequency,
            nu,
        )
        value = tail_norm(
            raw, args.temporal_modes, args.spatial_modes
        )
        if value > maximum_finite_tail[0]:
            maximum_finite_tail = (
                value,
                (harmonic, "sin" if is_sine else "cos", spatial_mode),
            )

    maximum_coarse = (0.0, None)
    refine = []
    max_t = args.temporal_modes + candidate_t
    max_x = args.spatial_modes + candidate_x
    for harmonic, is_sine, spatial_mode in basis_vectors(max_t, max_x):
        inside = (
            harmonic <= args.temporal_modes
            and spatial_mode <= args.spatial_modes
        )
        if inside:
            continue
        ahat = direct_ahat_column(
            centered,
            harmonic,
            is_sine,
            spatial_mode,
            frequency,
            nu,
        )
        core = core_part(
            ahat, args.temporal_modes, args.spatial_modes, allowed
        )
        cross_coarse = float(
            np.dot(inverse_column_norms, np.abs(core))
        )
        raw = direct_raw_tail_column(
            centered,
            harmonic,
            is_sine,
            spatial_mode,
            frequency,
            nu,
        )
        tail = tail_norm(raw, args.temporal_modes, args.spatial_modes)
        total = cross_coarse + tail
        metadata = (
            harmonic,
            "sin" if is_sine else "cos",
            spatial_mode,
            cross_coarse,
            tail,
        )
        if total > maximum_coarse[0]:
            maximum_coarse = (total, metadata)
        if total >= args.refine_threshold:
            refine.append((core, tail, metadata))

    maximum_refined = (0.0, None)
    for core, tail, metadata in refine:
        cross = float(np.sum(np.abs(inverse_state @ core)))
        total = cross + tail
        refined_metadata = metadata[:3] + (cross, tail)
        if total > maximum_refined[0]:
            maximum_refined = (total, refined_metadata)

    # Exhaust the part of the tail outside the finite coupling halo.  For
    # temporal input n >= T+Ut+40, every output has temporal harmonic at
    # least T+40.  The continuous maximum of
    # (a+z)/(a^2+z^2), z>=z0, occurs at max(z0,(sqrt(2)-1)*a).
    real_candidate = np.empty_like(centered.real)
    real_candidate[0] = centered[candidate_t].real
    real_candidate[1 : candidate_t + 1] = 2.0 * centered[candidate_t + 1 :].real
    real_candidate[candidate_t + 1 :] = -2.0 * centered[candidate_t + 1 :].imag
    spatial_l1 = np.sum(np.abs(real_candidate), axis=0)

    temporal_far_start = max_t + 40
    minimum_output_harmonic = temporal_far_start - candidate_t

    def inverse_continuous_sup(spatial_mode, minimum_harmonic):
        p = abs(nu * spatial_mode**4 - spatial_mode**2)
        z0 = frequency * minimum_harmonic
        z = max(z0, (np.sqrt(2.0) - 1.0) * p)
        return (p + z) / (p * p + z * z)

    maximum_temporal_far = (0.0, None)
    # Inputs with k > X+K are covered by the spatial-far estimate below.
    for input_space in range(1, max_x + 1):
        value = 0.0
        for u_space in range(1, candidate_x + 1):
            difference = abs(input_space - u_space)
            if difference:
                value += (
                    0.5
                    * difference
                    * inverse_continuous_sup(
                        difference, minimum_output_harmonic
                    )
                    * spatial_l1[u_space - 1]
                )
            total = input_space + u_space
            value += (
                0.5
                * total
                * inverse_continuous_sup(total, minimum_output_harmonic)
                * spatial_l1[u_space - 1]
            )
        if value > maximum_temporal_far[0]:
            maximum_temporal_far = (value, input_space)

    maximum_temporal_gap = (0.0, None)
    for harmonic, is_sine, spatial_mode in basis_vectors(
        temporal_far_start - 1, max_x
    ):
        if harmonic <= max_t:
            continue
        raw = direct_raw_tail_column(
            centered,
            harmonic,
            is_sine,
            spatial_mode,
            frequency,
            nu,
        )
        value = float(np.sum(np.abs(raw)))
        if value > maximum_temporal_gap[0]:
            maximum_temporal_gap = (
                value,
                (harmonic, "sin" if is_sine else "cos", spatial_mode),
            )

    def inverse_all_time_sup(spatial_mode):
        p = abs(nu * spatial_mode**4 - spatial_mode**2)
        z = (np.sqrt(2.0) - 1.0) * p
        return max(1.0 / p, (p + z) / (p * p + z * z))

    # For input k >= X+K+1 all product modes remain outside the core.  The
    # displayed maximum is scanned over a long finite range; monotonicity of
    # k/(nu*k^4-k^2) beyond this range gives the final analytic enclosure.
    maximum_spatial_far = (0.0, None)
    for input_space in range(max_x + 1, max_x + 2001):
        value = 0.0
        for u_space in range(1, candidate_x + 1):
            difference = input_space - u_space
            value += (
                0.5
                * difference
                * inverse_all_time_sup(difference)
                * spatial_l1[u_space - 1]
            )
            total = input_space + u_space
            value += (
                0.5
                * total
                * inverse_all_time_sup(total)
                * spatial_l1[u_space - 1]
            )
        if value > maximum_spatial_far[0]:
            maximum_spatial_far = (value, input_space)

    # Compute the fully de-aliased residual of the projected polynomial.
    output_t = 2 * candidate_t
    output_x = 2 * candidate_x
    node_count = 2 * output_t + 1
    phases = 2.0 * np.pi * np.arange(node_count) / node_count
    evaluation = np.exp(
        1j
        * phases[:, None]
        * np.arange(-candidate_t, candidate_t + 1)[None, :]
    )
    values = (evaluation @ centered).real
    nonlinear = np.empty((node_count, output_x))
    from diagnose_smooth_cap import (
        apply_raw_linear_inverse_real,
        apply_sinv_real,
        real_time_coefficients,
        spatial_product_derivative_matrix,
        temporal_derivative,
        temporal_projection,
    )

    for row_index, row in enumerate(values):
        padded = np.zeros(output_x)
        padded[:candidate_x] = row
        nonlinear[row_index] = (
            0.5
            * spatial_product_derivative_matrix(row, output_x)
            @ padded
        )
    residual = temporal_projection(output_t, phases) @ nonlinear
    candidate_real = real_time_coefficients(centered, candidate_t)
    linear = (
        frequency * temporal_derivative(candidate_t) @ candidate_real
        + (
            nu * np.arange(1, candidate_x + 1) ** 4
            - np.arange(1, candidate_x + 1) ** 2
        )[None, :]
        * candidate_real
    )
    residual[0, :candidate_x] += linear[0]
    residual[1 : candidate_t + 1, :candidate_x] += linear[
        1 : candidate_t + 1
    ]
    residual[
        output_t + 1 : output_t + candidate_t + 1, :candidate_x
    ] += linear[candidate_t + 1 :]
    smoothed_residual = apply_sinv_real(residual, frequency, nu)
    raw_residual = apply_raw_linear_inverse_real(residual, frequency, nu)
    residual_finite = float(
        np.sum(
            np.abs(
                inverse_state
                @ core_part(
                    smoothed_residual,
                    args.temporal_modes,
                    args.spatial_modes,
                    allowed,
                )
            )
        )
    )
    residual_tail = tail_norm(
        raw_residual, args.temporal_modes, args.spatial_modes
    )

    # Quadratic coefficient: ||B*S_c^-1*d_theta|| plus one half of
    # ||B*S_c^-1*d_x||.  Both are block diagonal between the finite block and
    # the raw-linear tail preconditioner.
    sinv_allowed = constants["sinv_allowed"]
    derivative_allowed = constants["derivative_allowed"]
    finite_theta = float(
        np.linalg.norm(
            inverse_state @ (sinv_allowed @ derivative_allowed), 1
        )
    )
    spatial_indices = np.tile(
        np.arange(1, args.spatial_modes + 1),
        2 * args.temporal_modes + 1,
    )[allowed]
    finite_space = float(
        np.linalg.norm(
            inverse_state
            @ (sinv_allowed * spatial_indices[None, :]),
            1,
        )
    )
    tail_theta = (1.0 + np.sqrt(2.0)) / (2.0 * frequency)
    # Continuous symbol bound for the tail space derivative.
    z_factor = np.sqrt(2.0) - 1.0
    tail_space_candidates = []
    for spatial_mode in range(1, args.spatial_modes + 1):
        p = abs(nu * spatial_mode**4 - spatial_mode**2)
        z0 = frequency * (args.temporal_modes + 1)
        z = max(z0, z_factor * p)
        tail_space_candidates.append(
            spatial_mode * (p + z) / (p * p + z * z)
        )
    spatial_mode = args.spatial_modes + 1
    p = abs(nu * spatial_mode**4 - spatial_mode**2)
    z = z_factor * p
    tail_space_candidates.append(
        spatial_mode * max(1.0 / p, (p + z) / (p * p + z * z))
    )
    tail_space = max(tail_space_candidates)
    quadratic_bound = max(finite_theta, tail_theta) + 0.5 * max(
        finite_space, tail_space
    )
    # The frequency is also a finite input column. Its state output includes
    # the candidate's spatial modes beyond K, which must not be omitted.
    full_candidate_real = real_time_coefficients(centered, candidate_t)
    frequency_tail = tail_norm(
        apply_raw_linear_inverse_real(
            temporal_derivative(candidate_t) @ full_candidate_real, frequency, nu
        ), args.temporal_modes, args.spatial_modes)

    print(f"dimension={len(inverse)}")
    print(f"finite_defect={finite_defect:.16e}")
    print(f"inverse_norm_1={np.linalg.norm(inverse, 1):.16e}")
    print(
        "maximum_finite_to_tail="
        f"{maximum_finite_tail[0]:.16e},mode={maximum_finite_tail[1]}"
    )
    print(
        "maximum_coarse_tail_column="
        f"{maximum_coarse[0]:.16e},mode={maximum_coarse[1]}"
    )
    print(f"refined_column_count={len(refine)}")
    print(
        "maximum_refined_tail_column="
        f"{maximum_refined[0]:.16e},mode={maximum_refined[1]}"
    )
    print(
        "scanned_tail_halo="
        f"time<= {max_t},space<= {max_x}"
    )
    print(f"frequency={constants['frequency']:.16e}")
    print(f"nu={constants['nu']:.16e}")
    print(
        "maximum_temporal_gap="
        f"{maximum_temporal_gap[0]:.16e},mode={maximum_temporal_gap[1]}"
    )
    print(
        "maximum_temporal_far="
        f"{maximum_temporal_far[0]:.16e},space={maximum_temporal_far[1]},"
        f"input_time>={temporal_far_start}"
    )
    print(
        "maximum_spatial_far="
        f"{maximum_spatial_far[0]:.16e},space={maximum_spatial_far[1]}"
    )
    print(f"residual_finite={residual_finite:.16e}")
    print(f"residual_tail={residual_tail:.16e}")
    print(f"quadratic_bound={quadratic_bound:.16e}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        savemat(
            args.output,
            {
                "temporal_modes": args.temporal_modes,
                "spatial_modes": args.spatial_modes,
                "candidate_temporal_modes": candidate_t,
                "candidate_spatial_modes": candidate_x,
                "dimension": len(inverse),
                "finite_defect_mid": finite_defect,
                "finite_operator_norm_1": np.linalg.norm(finite_operator, 1),
                "inverse_norm_1": np.linalg.norm(inverse, 1),
                "maximum_finite_to_tail_mid": maximum_finite_tail[0],
                "frequency_to_tail_mid": frequency_tail,
                "maximum_coarse_tail_column_mid": maximum_coarse[0],
                "maximum_refined_tail_column_mid": maximum_refined[0],
                "refine_threshold": args.refine_threshold,
                "maximum_temporal_gap_mid": maximum_temporal_gap[0],
                "maximum_temporal_far_mid": maximum_temporal_far[0],
                "maximum_spatial_far_mid": maximum_spatial_far[0],
                "residual_finite_mid": residual_finite,
                "residual_tail_mid": residual_tail,
                "quadratic_bound_mid": quadratic_bound,
                "finite_theta_mid": finite_theta,
                "finite_space_mid": finite_space,
                "tail_theta_mid": tail_theta,
                "tail_space_mid": tail_space,
                "frequency_mid": frequency,
                "nu_mid": nu,
                "projected_velocity_time_fourier": physical,
                "source_frequency": np.squeeze(data["frequency"]),
                "source_interval_length": ell,
                # This finite-rank operator is part of the mathematical
                # certificate.  It is regenerated rather than committed.
                "finite_inverse": inverse,
            },
            do_compression=True,
        )


if __name__ == "__main__":
    main()
