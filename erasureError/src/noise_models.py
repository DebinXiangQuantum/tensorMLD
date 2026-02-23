# ============================================================================ #
# Noise Models for Erasure Error Tensor Network Decoder                        #
# ============================================================================ #
"""
This module implements noise models that support erasure errors for tensor
network decoding.

Mathematical Background (from erasureerror.md):
-----------------------------------------------
For erasure errors, the qubit location is known (heralded) but the Pauli type
(X or Z) is unknown. This leads to:

1. Standard Error: p = physical_error_rate (small probability)
2. Erasure Error: p = 0.5 (maximum uncertainty about error type)

The weight β in the tensor network is defined as:
    β = (1/2) * ln((1-p)/p)

For erasure (p=0.5):
    β = (1/2) * ln(1) = 0

This causes the energy tensor to collapse to identity:
    T(e) = [1, 1] for both e=0 and e=1

This corresponds to "broken bonds" in the spin-glass model, where there is
zero energy penalty for assigning an error to an erased qubit.
"""

from typing import Any, Optional, Union
import numpy as np
from quimb import oset
from quimb.tensor import TensorNetwork, Tensor


def factorized_noise_model_with_erasure(
        error_indices: list[str],
        error_probabilities: Union[list[float], np.ndarray],
        erasure_mask: Optional[Union[list[bool], np.ndarray]] = None,
        tensors_tags: Optional[list[str]] = None,
        debug: bool = False) -> TensorNetwork:
    """
    Construct a factorized (product state) noise model with erasure support.

    For standard errors, the tensor is [1-p, p].
    For erased qubits (erasure_mask[i] = True), the tensor is [0.5, 0.5],
    representing maximum uncertainty.

    Args:
        error_indices: List of error index names.
        error_probabilities: List or array of error probabilities for each
                           error index.
        erasure_mask: Boolean mask indicating which qubits are erased.
                     If None, no qubits are treated as erased.
        tensors_tags: List of tags for each tensor.
        debug: If True, print debug information.

    Returns:
        TensorNetwork: The tensor network representing the noise model.
    """
    assert len(error_indices) == len(error_probabilities), \
        "Length of error_indices and error_probabilities must match."

    if isinstance(error_probabilities, np.ndarray):
        assert error_probabilities.ndim == 1 and len(error_probabilities) == len(
            error_indices
        ), "error_probabilities must be a 1D array with length matching error_indices."
        error_probabilities = error_probabilities.tolist()
    elif isinstance(error_probabilities, list):
        assert all(isinstance(p, (float, int)) for p in error_probabilities), \
            "error_probabilities must be a list of floats or ints."
    else:
        raise TypeError("error_probabilities must be a list or numpy array.")

    assert all(p >= 0 and p <= 1 for p in error_probabilities), \
        "All error probabilities must be in the range [0, 1]."
    assert all(isinstance(ind, str) for ind in error_indices), \
        "All error indices must be strings."

    # Handle erasure mask
    if erasure_mask is None:
        erasure_mask = [False] * len(error_indices)
    elif isinstance(erasure_mask, np.ndarray):
        erasure_mask = erasure_mask.tolist()

    assert len(erasure_mask) == len(error_indices), \
        "erasure_mask must have the same length as error_indices."

    tensors = []

    if tensors_tags is None:
        tensors_tags = ["NOISE"] * len(error_indices)

    num_erased = sum(erasure_mask)
    if debug:
        print(f"[DEBUG] Creating noise model with {len(error_indices)} errors, "
              f"{num_erased} erased qubits")

    for i, (ei, eprob, etag) in enumerate(zip(error_indices, error_probabilities,
                                              tensors_tags)):
        if erasure_mask[i]:
            # Erased qubit: maximum uncertainty (p = 0.5)
            # Tensor is [0.5, 0.5], representing equal probability of error or no error
            data = np.array([0.5, 0.5])
            if debug:
                print(f"[DEBUG] Error index {ei}: ERASED (tensor=[0.5, 0.5])")
        else:
            # Standard error
            data = np.array([1.0 - eprob, eprob])
            if debug and i < 5:  # Only print first few for brevity
                print(f"[DEBUG] Error index {ei}: p={eprob:.4f} "
                      f"(tensor=[{1.0-eprob:.4f}, {eprob:.4f}])")

        tensors.append(
            Tensor(
                data=data,
                inds=(ei,),
                tags=oset([etag]),
            ))

    return TensorNetwork(tensors)


def create_erasure_noise_model(
        base_error_rate: float,
        erasure_rate: float,
        num_errors: int,
        error_indices: Optional[list[str]] = None,
        random_seed: Optional[int] = None,
        debug: bool = False) -> tuple[TensorNetwork, np.ndarray]:
    """
    Create a noise model with random erasures based on erasure rate.

    This function models a scenario where:
    1. Each qubit has a base depolarizing error rate
    2. Additionally, each qubit has a probability of being "erased" (heralded)

    Args:
        base_error_rate: The base depolarizing error probability for non-erased qubits.
        erasure_rate: The probability of a qubit being erased (heralded failure).
        num_errors: Number of error locations.
        error_indices: List of error index names. If None, defaults to [e_0, e_1, ...].
        random_seed: Random seed for reproducibility.
        debug: If True, print debug information.

    Returns:
        Tuple of (TensorNetwork, erasure_mask) where:
            - TensorNetwork is the noise model
            - erasure_mask is a boolean array indicating erased positions
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if error_indices is None:
        error_indices = [f"e_{i}" for i in range(num_errors)]

    # Generate erasure mask based on erasure rate
    erasure_mask = np.random.rand(num_errors) < erasure_rate

    # Base error probabilities for all qubits
    error_probabilities = [base_error_rate] * num_errors

    if debug:
        print(f"[DEBUG] Creating noise model: base_error_rate={base_error_rate}, "
              f"erasure_rate={erasure_rate}, num_errors={num_errors}")
        print(f"[DEBUG] Number of erased qubits: {np.sum(erasure_mask)}")

    noise_model = factorized_noise_model_with_erasure(
        error_indices=error_indices,
        error_probabilities=error_probabilities,
        erasure_mask=erasure_mask,
        debug=debug
    )

    return noise_model, erasure_mask


def create_deterministic_erasure_noise_model(
        error_probabilities: Union[list[float], np.ndarray],
        erased_indices: list[int],
        error_indices: Optional[list[str]] = None,
        debug: bool = False) -> TensorNetwork:
    """
    Create a noise model with specific qubits marked as erased.

    Args:
        error_probabilities: Error probabilities for each location.
        erased_indices: List of indices to mark as erased.
        error_indices: List of error index names.
        debug: If True, print debug information.

    Returns:
        TensorNetwork: The noise model with specified erasures.
    """
    if isinstance(error_probabilities, np.ndarray):
        error_probabilities = error_probabilities.tolist()

    num_errors = len(error_probabilities)

    if error_indices is None:
        error_indices = [f"e_{i}" for i in range(num_errors)]

    # Create erasure mask from specified indices
    erasure_mask = [False] * num_errors
    for idx in erased_indices:
        if 0 <= idx < num_errors:
            erasure_mask[idx] = True

    if debug:
        print(f"[DEBUG] Creating noise model with {len(erased_indices)} "
              f"deterministically erased qubits at indices: {erased_indices}")

    return factorized_noise_model_with_erasure(
        error_indices=error_indices,
        error_probabilities=error_probabilities,
        erasure_mask=erasure_mask,
        debug=debug
    )
