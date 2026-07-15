# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Utility functions for differential privacy.

MODERN L2-based DP Mechanisms:
- L2 clipping + Gaussian noise: compute_clip_model_update + add_localdp_fixed_gaussian_noise_to_params
- L2 clipping + Laplace noise: compute_clip_model_update + add_localdp_l2laplace_noise_to_params

All functions use L2 norm for consistency and better privacy guarantees.
"""


from logging import WARNING, INFO
from typing import Optional

import numpy as np

from flwr.common.typing import NDArrays, Parameters
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays


def get_norm(input_arrays: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    flattened = np.concatenate([array.flatten() for array in input_arrays])
    return float(np.linalg.norm(flattened))


def add_gaussian_noise_inplace(input_arrays: NDArrays, std_dev: float) -> None:
    """Add Gaussian noise to each element of the input arrays."""
    for array in input_arrays:
        array += np.random.normal(0, std_dev, array.shape)


def clip_inputs_inplace(input_arrays: NDArrays, clipping_norm: float) -> None:
    """Clip model update based on the L2 clipping norm in-place.

    FlatClip method of the paper: https://arxiv.org/abs/1710.06963
    """
    input_norm = get_norm(input_arrays)
    if input_norm <= 1e-12:
        scaling_factor = 1.0
    else:
        scaling_factor = min(1.0, clipping_norm / input_norm)
    for array in input_arrays:
        array *= scaling_factor


def compute_stdv(
    noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    Paper: https://arxiv.org/abs/1710.06963
    """
    from flwr.common.logger import log
    from logging import WARNING

    if num_sampled_clients <= 0:
        log(WARNING, "num_sampled_clients <= 0, falling back to 1.")
        num_sampled_clients = 1
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)


def compute_clip_model_update(
    param1: NDArrays, param2: NDArrays, clipping_norm: float
) -> None:
    """Compute model update (param1 - param2) and clip it using L2 norm.

    Then add the clipped value to param1."""
    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2)]
    clip_inputs_inplace(model_update, clipping_norm)

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]


def adaptive_clip_inputs_inplace(input_arrays: NDArrays, clipping_norm: float) -> bool:
    """Clip model update based on the L2 clipping norm in-place.

    It returns true if scaling_factor < 1 which is used for norm_bit
    FlatClip method of the paper: https://arxiv.org/abs/1710.06963
    """
    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    for array in input_arrays:
        array *= scaling_factor
    return scaling_factor < 1


def compute_adaptive_clip_model_update(
    param1: NDArrays, param2: NDArrays, clipping_norm: float
) -> bool:
    """Compute model update, clip it using L2 norm, then add the clipped value to param1.

    model update = param1 - param2
    Return the norm_bit
    """
    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2)]
    norm_bit = adaptive_clip_inputs_inplace(model_update, clipping_norm)

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]

    return norm_bit


def add_gaussian_noise_to_params(
    model_params: Parameters,
    noise_multiplier: float,
    clipping_norm: float,
    num_sampled_clients: int,
) -> Parameters:
    """Add gaussian noise to model parameters."""
    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients),
    )
    return ndarrays_to_parameters(model_params_ndarrays)


def compute_adaptive_noise_params(
    noise_multiplier: float,
    num_sampled_clients: float,
    clipped_count_stddev: Optional[float],
) -> tuple[float, float]:
    """Compute noising parameters for the adaptive clipping.

    Paper: https://arxiv.org/abs/1905.03871
    """
    if noise_multiplier > 0:
        if clipped_count_stddev is None:
            clipped_count_stddev = num_sampled_clients / 20
        if noise_multiplier >= 2 * clipped_count_stddev:
            raise ValueError(
                f"If not specified, `clipped_count_stddev` is set to "
                f"`num_sampled_clients`/20 by default. This value "
                f"({num_sampled_clients / 20}) is too low to achieve the "
                f"desired effective `noise_multiplier` ({noise_multiplier}). "
                f"Consider increasing `clipped_count_stddev` or decreasing "
                f"`noise_multiplier`."
            )
        noise_multiplier_value = (
            noise_multiplier ** (-2) - (2 * clipped_count_stddev) ** (-2)
        ) ** -0.5

    else:
        noise_multiplier_value = 0.0

    return noise_multiplier_value, clipped_count_stddev


def compute_adaptive_clip_noise_params(
    noise_multiplier: float,
    clipping_norm: float,
    num_sampled_clients: int,
    clipped_count_stddev: Optional[float],
) -> tuple[float, float]:
    """Compute global noise params, clipped count noise for DP with adaptive clipping.

    Paper: https://arxiv.org/abs/1905.03871
    """
    noise_multiplier_value, clipped_count_stddev = compute_adaptive_noise_params(
        noise_multiplier, num_sampled_clients, clipped_count_stddev
    )

    stdv = compute_stdv(noise_multiplier_value, clipping_norm, num_sampled_clients)

    return stdv, clipped_count_stddev


def add_adaptive_noise_to_params(
    model_params: Parameters,
    noise_multiplier: float,
    clipping_norm: float,
    num_sampled_clients: int,
    clipped_count_stddev: Optional[float] = None,
) -> Parameters:
    """Add adaptive gaussian noise to model parameters.

    Paper: https://arxiv.org/abs/1905.03871
    """
    model_params_ndarrays = parameters_to_ndarrays(model_params)

    stdv, _ = compute_adaptive_clip_noise_params(
        noise_multiplier, clipping_norm, num_sampled_clients, clipped_count_stddev
    )

    add_gaussian_noise_inplace(model_params_ndarrays, stdv)

    return ndarrays_to_parameters(model_params_ndarrays)


def add_localdp_fixed_gaussian_noise_to_params(
    model_params: NDArrays, noise_stddev: float
) -> NDArrays:
    """Add Fixed Gaussian DP noise (sigma = C x noise_multiplier) to model parameters.

    Args:
        model_params: Model parameters as NDArrays
        noise_stddev: Standard deviation of Gaussian noise (sigma = C x noise_multiplier)

    Returns:
        Model parameters with added noise
    """
    # Apply noise to each parameter array
    for param_array in model_params:
        noise = np.random.normal(0, noise_stddev, param_array.shape)
        param_array += noise.astype(param_array.dtype)

    return model_params


def add_localdp_l2laplace_noise_to_params(
    model_params: NDArrays, scale: float
) -> NDArrays:
    """Add L2-compatible Laplace DP noise to model parameters.

    This function adds coordinate-wise Laplace noise that is compatible with L2 clipping.
    The noise scale is calculated as: b = C / epsilon_round

    Args:
        model_params: Model parameters as NDArrays
        scale: Laplace noise scale parameter (b = C / epsilon_round)

    Returns:
        Model parameters with added Laplace noise
    """
    # Apply coordinate-wise Laplace noise to each parameter array
    for param_array in model_params:
        noise = np.random.laplace(0, scale, param_array.shape)
        param_array += noise.astype(param_array.dtype)

    return model_params


def find_sigma_for_target_epsilon(
    target_epsilon: float,
    num_rounds: int,
    q: float = 1.0,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
    max_iterations: int = 1000
) -> float:
    """
    Solve for the noise multiplier (sigma / C) with an RDP accountant so the
    total privacy cost approaches the target epsilon after all rounds.

    Args:
        target_epsilon: Target total privacy budget.
        num_rounds: Number of training rounds.
        q: Sampling rate, normally 1.0 for client-level DP.
        max_grad_norm: Clipping norm C.
        delta: Gaussian mechanism delta.
        max_iterations: Maximum binary-search iterations.

    Returns:
        Calibrated noise multiplier (sigma / C).
    """
    from flwr.common.logger import log

    try:
        from opacus.accountants import RDPAccountant
    except ImportError:
        log(WARNING, "Opacus not available, using fallback calculation")
        # Conservative fallback when Opacus is unavailable.
        return target_epsilon / (2 * num_rounds * q * np.sqrt(2 * np.log(1.25 / delta)))

    # Solve with binary search.
    low, high = 0.1, 10.0

    for iteration in range(max_iterations):
        mid = (low + high) / 2

        # Simulate all rounds with the candidate multiplier.
        accountant = RDPAccountant()

        for round_idx in range(num_rounds):
            accountant.step(
                noise_multiplier=mid,
                sample_rate=q
            )

        computed_epsilon = accountant.get_epsilon(delta)

        if abs(computed_epsilon - target_epsilon) < 1e-6:
            log(INFO, f"Converged after {iteration + 1} iterations")
            break

        if computed_epsilon > target_epsilon:
            low = mid
        else:
            high = mid

    final_noise_multiplier = (low + high) / 2
    log(INFO, f"Found noise_multiplier: {final_noise_multiplier:.6f} for target_epsilon: {target_epsilon}")

    return final_noise_multiplier
