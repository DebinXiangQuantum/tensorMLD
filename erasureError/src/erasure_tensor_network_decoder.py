# ============================================================================ #
# Erasure Error Tensor Network Decoder                                         #
# Extension of tensor network decoder for erasure errors                       #
# ============================================================================ #
"""
This module implements a tensor network decoder that supports erasure errors.

The implementation is based on the mathematical framework from TensorMLD papers,
extended to handle erasure errors where the error location is known (heralded)
but the Pauli type is unknown.

Key Mathematical Concepts:
--------------------------
1. For standard depolarizing errors:
   - Error probability p is small
   - Weight β = (1/2) * ln((1-p)/p)
   - Tensor T = [sqrt(1-p), sqrt(p)]

2. For erasure errors:
   - Error probability p = 0.5 (maximum uncertainty)
   - Weight β = 0
   - Tensor T = [0.5, 0.5] (identity-like, "broken bond")

This implementation supports:
- Circuit-level noise with erasure errors
- Batch decoding for efficiency
- Integration with stim for syndrome sampling
"""

from typing import Optional, Any, Union
import numpy as np
import numpy.typing as npt
from quimb.tensor import TensorNetwork, Tensor
from quimb import oset

try:
    from .noise_models import factorized_noise_model_with_erasure
except ImportError:
    from noise_models import factorized_noise_model_with_erasure


class ErasureTensorNetworkDecoder:
    """
    Tensor network decoder supporting erasure errors for MLD.

    This decoder constructs a tensor network representing the code structure
    and contracts it to compute the probability of logical errors given
    the syndrome and known erasure locations.

    Attributes:
        H: The parity check matrix (detectors x errors).
        logical_obs: The logical observable matrix.
        error_indices: List of error index names.
        check_indices: List of check/syndrome index names.
        erasure_mask: Boolean mask indicating erased positions.
        debug: Whether to print debug information.
    """

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[list[float], np.ndarray],
        erasure_mask: Optional[Union[list[bool], np.ndarray]] = None,
        check_inds: Optional[list[str]] = None,
        error_inds: Optional[list[str]] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the erasure-aware tensor network decoder.

        Args:
            H: Parity check matrix. Shape (num_checks, num_errors).
            logical_obs: Logical observable matrix. Shape (1, num_errors).
            error_probabilities: Error probabilities for each error location.
            erasure_mask: Boolean mask indicating which qubits are erased.
                         If None, no qubits are treated as erased.
            check_inds: Names for check indices. Defaults to [s_0, s_1, ...].
            error_inds: Names for error indices. Defaults to [e_0, e_1, ...].
            debug: If True, print debug information.
        """
        self.debug = debug
        self.H = H.copy()
        self.logical_obs = logical_obs.copy()

        num_checks, num_errors = H.shape

        if self.debug:
            print(f"[DEBUG] Initializing ErasureTensorNetworkDecoder")
            print(f"[DEBUG] Parity check matrix shape: {H.shape}")
            print(f"[DEBUG] Logical observable shape: {logical_obs.shape}")

        # Setup indices
        if check_inds is None:
            self.check_inds = [f"s_{j}" for j in range(num_checks)]
        else:
            assert len(check_inds) == num_checks
            self.check_inds = check_inds

        if error_inds is None:
            self.error_inds = [f"e_{j}" for j in range(num_errors)]
        else:
            assert len(error_inds) == num_errors
            self.error_inds = error_inds

        self.logical_obs_ind = "obs"  # Open logical index

        # Setup error probabilities
        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.tolist()
        else:
            self.error_probs = list(error_probabilities)

        # Setup erasure mask
        if erasure_mask is None:
            self.erasure_mask = np.zeros(num_errors, dtype=bool)
        elif isinstance(erasure_mask, list):
            self.erasure_mask = np.array(erasure_mask, dtype=bool)
        else:
            self.erasure_mask = erasure_mask.copy()

        if self.debug:
            num_erased = np.sum(self.erasure_mask)
            print(f"[DEBUG] Number of erased qubits: {num_erased}/{num_errors}")

        # Build tensor network
        self._build_tensor_network()

    def _build_tensor_network(self) -> None:
        """Build the tensor network for decoding."""
        if self.debug:
            print("[DEBUG] Building tensor network...")

        # Build parity check tensor network (Hadamard matrices)
        self.code_tn = self._tensor_network_from_parity_check(
            self.H,
            col_inds=self.error_inds,
            row_inds=self.check_inds,
        )

        # Build logical observable tensor network
        self.logical_tn = self._build_logical_tn()

        # Build noise model tensor network
        self.noise_tn = factorized_noise_model_with_erasure(
            error_indices=self.error_inds,
            error_probabilities=self.error_probs,
            erasure_mask=self.erasure_mask,
            debug=self.debug
        )

        if self.debug:
            print(f"[DEBUG] Code TN tensors: {len(self.code_tn.tensors)}")
            print(f"[DEBUG] Logical TN tensors: {len(self.logical_tn.tensors)}")
            print(f"[DEBUG] Noise TN tensors: {len(self.noise_tn.tensors)}")

    def _tensor_network_from_parity_check(
        self,
        parity_check_matrix: npt.NDArray[Any],
        row_inds: list[str],
        col_inds: list[str],
        tags: Optional[list[str]] = None,
    ) -> TensorNetwork:
        """
        Build a sparse tensor-network from a parity-check matrix.

        For each non-zero entry H[i,j], add a Hadamard tensor connecting
        row index i and column index j.
        """
        hadamard = np.array([[1.0, 1.0], [1.0, -1.0]])
        rows, cols = np.nonzero(parity_check_matrix)

        tensors = []
        for i, j in zip(rows, cols):
            tensor_tags = oset([tags[i]] if tags is not None else [])
            tensors.append(
                Tensor(
                    data=hadamard,
                    inds=(row_inds[i], col_inds[j]),
                    tags=tensor_tags,
                )
            )

        return TensorNetwork(tensors)

    def _build_logical_tn(self) -> TensorNetwork:
        """Build tensor network for logical observable."""
        # Logical observable connects error indices to logical output
        hadamard = np.array([[1.0, 1.0], [1.0, -1.0]])

        # For each non-zero in logical_obs, add Hadamard tensor
        rows, cols = np.nonzero(self.logical_obs)

        tensors = []
        logical_inds = ["l_0"]  # Intermediate logical index

        for i, j in zip(rows, cols):
            tensors.append(
                Tensor(
                    data=hadamard,
                    inds=(logical_inds[i], self.error_inds[j]),
                    tags=oset(["LOGICAL"]),
                )
            )

        # Add final Hadamard for logical observable output
        tensors.append(
            Tensor(
                data=hadamard,
                inds=(logical_inds[0], self.logical_obs_ind),
                tags=oset(["LOGICAL_OBS"]),
            )
        )

        return TensorNetwork(tensors)

    def _build_syndrome_tn(self, syndrome: list[float]) -> TensorNetwork:
        """
        Build tensor network for syndrome.

        Args:
            syndrome: List of syndrome values (probabilities that each
                     check was triggered).
        """
        minus = np.array([1.0, -1.0])
        plus = np.array([1.0, 1.0])

        tensors = []
        for i, (sind, sval) in enumerate(zip(self.check_inds, syndrome)):
            # sval is the probability that the syndrome is 1
            # Tensor is sval * [1, -1] + (1-sval) * [1, 1]
            data = sval * minus + (1.0 - sval) * plus
            tensors.append(
                Tensor(
                    data=data,
                    inds=(sind,),
                    tags=oset([f"SYN_{i}", "SYNDROME"]),
                )
            )

        return TensorNetwork(tensors)

    def decode(self, syndrome: list[float]) -> dict:
        """
        Decode a single syndrome.

        Args:
            syndrome: List of syndrome values. Each value is the probability
                     that the corresponding check triggered (0.0 or 1.0 for
                     hard decision, or soft probabilities).

        Returns:
            Dictionary with:
                - 'logical_error_prob': Probability of logical error
                - 'converged': Always True for exact decoder
                - 'result': Same as logical_error_prob for compatibility
        """
        assert len(syndrome) == len(self.check_inds), \
            f"Syndrome length {len(syndrome)} != num_checks {len(self.check_inds)}"

        if self.debug:
            triggered = sum(1 for s in syndrome if s > 0.5)
            print(f"[DEBUG] Decoding syndrome with {triggered} triggered checks")

        # Build syndrome tensor network
        syndrome_tn = self._build_syndrome_tn(syndrome)

        # Combine all tensor networks
        full_tn = TensorNetwork()
        full_tn = full_tn.combine(self.code_tn, virtual=True)
        full_tn = full_tn.combine(self.logical_tn, virtual=True)
        full_tn = full_tn.combine(self.noise_tn, virtual=True)
        full_tn = full_tn.combine(syndrome_tn, virtual=True)

        # Contract tensor network, keeping logical observable index open
        try:
            result = full_tn.contract(output_inds=(self.logical_obs_ind,))
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Contraction failed: {e}")
            # Return 0.5 on failure (maximum uncertainty)
            return {
                'logical_error_prob': 0.5,
                'converged': False,
                'result': [0.5]
            }

        # Extract logical error probability
        # result[0] = amplitude for logical = 0
        # result[1] = amplitude for logical = 1
        result_array = np.array(result.data)

        if self.debug:
            print(f"[DEBUG] Contraction result: {result_array}")

        # Normalize to get probability
        total = result_array[0] + result_array[1]
        if abs(total) < 1e-10:
            # Degenerate case
            logical_error_prob = 0.5
        else:
            logical_error_prob = float(result_array[1] / total)

        return {
            'logical_error_prob': logical_error_prob,
            'converged': True,
            'result': [logical_error_prob]
        }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> list[dict]:
        """
        Decode a batch of syndromes.

        Args:
            syndromes: 2D array of shape (num_shots, num_checks).

        Returns:
            List of decoding results.
        """
        results = []
        for i, syndrome in enumerate(syndromes):
            if self.debug and i < 3:
                print(f"[DEBUG] Decoding shot {i}")
            syn_list = [float(s) for s in syndrome]
            results.append(self.decode(syn_list))
        return results

    def update_erasure_mask(self, new_erasure_mask: Union[list[bool], np.ndarray]) -> None:
        """
        Update the erasure mask and rebuild the noise model.

        Args:
            new_erasure_mask: New boolean mask indicating erased positions.
        """
        if isinstance(new_erasure_mask, list):
            self.erasure_mask = np.array(new_erasure_mask, dtype=bool)
        else:
            self.erasure_mask = new_erasure_mask.copy()

        if self.debug:
            print(f"[DEBUG] Updating erasure mask. New erased count: "
                  f"{np.sum(self.erasure_mask)}")

        # Rebuild noise tensor network
        self.noise_tn = factorized_noise_model_with_erasure(
            error_indices=self.error_inds,
            error_probabilities=self.error_probs,
            erasure_mask=self.erasure_mask,
            debug=self.debug
        )

    def set_error_probabilities(
        self,
        error_probabilities: Union[list[float], np.ndarray]
    ) -> None:
        """
        Update error probabilities and rebuild noise model.

        Args:
            error_probabilities: New error probabilities.
        """
        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.tolist()
        else:
            self.error_probs = list(error_probabilities)

        # Rebuild noise tensor network
        self.noise_tn = factorized_noise_model_with_erasure(
            error_indices=self.error_inds,
            error_probabilities=self.error_probs,
            erasure_mask=self.erasure_mask,
            debug=self.debug
        )
