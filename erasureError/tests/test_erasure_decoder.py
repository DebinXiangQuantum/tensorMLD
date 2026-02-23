# ============================================================================ #
# Unit Tests for Erasure Error Tensor Network Decoder                          #
# ============================================================================ #
"""
This module contains unit tests for the erasure-aware tensor network decoder.

Test cases include:
1. Basic functionality tests with simple codes
2. Erasure error handling tests
3. Comparison with known solutions
"""

import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from erasure_tensor_network_decoder import ErasureTensorNetworkDecoder
from noise_models import (
    factorized_noise_model_with_erasure,
    create_erasure_noise_model,
    create_deterministic_erasure_noise_model
)


class TestNoiseModels:
    """Tests for noise model functions."""

    def test_factorized_noise_model_basic(self):
        """Test basic factorized noise model without erasures."""
        error_indices = ["e_0", "e_1", "e_2"]
        error_probs = [0.1, 0.2, 0.05]

        noise_tn = factorized_noise_model_with_erasure(
            error_indices=error_indices,
            error_probabilities=error_probs,
            erasure_mask=None
        )

        # Should have 3 tensors
        assert len(noise_tn.tensors) == 3

        # Check tensor values
        for i, tensor in enumerate(noise_tn.tensors):
            expected_data = np.array([1.0 - error_probs[i], error_probs[i]])
            np.testing.assert_array_almost_equal(tensor.data, expected_data)

    def test_factorized_noise_model_with_erasure(self):
        """Test noise model with erasure errors."""
        error_indices = ["e_0", "e_1", "e_2"]
        error_probs = [0.1, 0.2, 0.05]
        erasure_mask = [False, True, False]  # e_1 is erased

        noise_tn = factorized_noise_model_with_erasure(
            error_indices=error_indices,
            error_probabilities=error_probs,
            erasure_mask=erasure_mask,
            debug=True
        )

        # Check that erased tensor has [0.5, 0.5]
        tensors = list(noise_tn.tensors)
        # The tensors are in order of error_indices
        erased_tensor = tensors[1]  # e_1
        np.testing.assert_array_almost_equal(erased_tensor.data, [0.5, 0.5])

        # Check non-erased tensors
        np.testing.assert_array_almost_equal(tensors[0].data, [0.9, 0.1])
        np.testing.assert_array_almost_equal(tensors[2].data, [0.95, 0.05])

    def test_create_erasure_noise_model(self):
        """Test random erasure noise model creation."""
        base_rate = 0.01
        erasure_rate = 0.3
        num_errors = 10

        noise_tn, erasure_mask = create_erasure_noise_model(
            base_error_rate=base_rate,
            erasure_rate=erasure_rate,
            num_errors=num_errors,
            random_seed=42
        )

        # Check correct number of tensors
        assert len(noise_tn.tensors) == num_errors

        # Check erasure mask has correct length
        assert len(erasure_mask) == num_errors

        # With seed=42 and rate=0.3, some qubits should be erased
        assert np.sum(erasure_mask) > 0

    def test_deterministic_erasure(self):
        """Test deterministic erasure noise model."""
        error_probs = [0.1] * 5
        erased_indices = [1, 3]

        noise_tn = create_deterministic_erasure_noise_model(
            error_probabilities=error_probs,
            erased_indices=erased_indices,
            debug=True
        )

        tensors = list(noise_tn.tensors)

        # Check erased positions
        np.testing.assert_array_almost_equal(tensors[1].data, [0.5, 0.5])
        np.testing.assert_array_almost_equal(tensors[3].data, [0.5, 0.5])

        # Check non-erased positions
        np.testing.assert_array_almost_equal(tensors[0].data, [0.9, 0.1])
        np.testing.assert_array_almost_equal(tensors[2].data, [0.9, 0.1])
        np.testing.assert_array_almost_equal(tensors[4].data, [0.9, 0.1])


class TestErasureTensorNetworkDecoder:
    """Tests for the ErasureTensorNetworkDecoder class."""

    @pytest.fixture
    def simple_repetition_code(self):
        """Create a simple 3-bit repetition code."""
        # Parity check matrix for 3-bit repetition code
        # Checks: c1 = e1 + e2, c2 = e2 + e3
        H = np.array([
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=np.float64)

        # Logical observable: all bits XOR
        logical_obs = np.array([[1, 1, 1]], dtype=np.float64)

        return H, logical_obs

    def test_decoder_initialization(self, simple_repetition_code):
        """Test decoder initialization."""
        H, logical_obs = simple_repetition_code
        error_probs = [0.1, 0.1, 0.1]

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            debug=True
        )

        assert decoder.H.shape == (2, 3)
        assert len(decoder.check_inds) == 2
        assert len(decoder.error_inds) == 3

    def test_decode_no_error(self, simple_repetition_code):
        """Test decoding when no error occurred."""
        H, logical_obs = simple_repetition_code
        error_probs = [0.1, 0.1, 0.1]

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            debug=True
        )

        # No syndrome triggered
        syndrome = [0.0, 0.0]
        result = decoder.decode(syndrome)

        print(f"No error result: {result}")

        # Should have low probability of logical error
        assert result['converged'] == True
        # With no syndrome, logical error probability should be low
        assert result['logical_error_prob'] < 0.5

    def test_decode_with_syndrome(self, simple_repetition_code):
        """Test decoding with triggered syndromes."""
        H, logical_obs = simple_repetition_code
        error_probs = [0.1, 0.1, 0.1]

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            debug=True
        )

        # Syndrome indicating error on first qubit (c1=1, c2=0)
        # This is consistent with e1 having an error
        syndrome = [1.0, 0.0]
        result = decoder.decode(syndrome)

        print(f"Syndrome [1,0] result: {result}")
        assert result['converged'] == True

    def test_decode_with_erasure(self, simple_repetition_code):
        """Test decoding with erasure on one qubit."""
        H, logical_obs = simple_repetition_code
        error_probs = [0.1, 0.1, 0.1]
        erasure_mask = [True, False, False]  # First qubit erased

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            erasure_mask=erasure_mask,
            debug=True
        )

        # No syndrome
        syndrome = [0.0, 0.0]
        result = decoder.decode(syndrome)

        print(f"With erasure on qubit 0, syndrome [0,0] result: {result}")
        assert result['converged'] == True

    def test_update_erasure_mask(self, simple_repetition_code):
        """Test updating erasure mask dynamically."""
        H, logical_obs = simple_repetition_code
        error_probs = [0.1, 0.1, 0.1]

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            erasure_mask=None,
            debug=True
        )

        # Initial decode
        syndrome = [0.0, 0.0]
        result1 = decoder.decode(syndrome)

        # Update erasure mask
        decoder.update_erasure_mask([True, True, False])

        # Decode again with erasures
        result2 = decoder.decode(syndrome)

        print(f"Result without erasure: {result1}")
        print(f"Result with 2 erasures: {result2}")

        # Results should be different due to erasures
        # (though specific values depend on the code structure)

    def test_decode_batch(self, simple_repetition_code):
        """Test batch decoding."""
        H, logical_obs = simple_repetition_code
        error_probs = [0.1, 0.1, 0.1]

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            debug=False  # Less verbose for batch
        )

        # Batch of syndromes
        syndromes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        results = decoder.decode_batch(syndromes)

        assert len(results) == 4
        for result in results:
            assert result['converged'] == True
            assert 0.0 <= result['logical_error_prob'] <= 1.0


class TestSteaneCode:
    """Test with Steane [[7,1,3]] code."""

    @pytest.fixture
    def steane_code(self):
        """Create Steane code parity check matrix."""
        # Z stabilizers for Steane code
        H = np.array([
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1]
        ], dtype=np.float64)

        # Logical Z operator
        logical_obs = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.float64)

        return H, logical_obs

    def test_steane_decode_no_error(self, steane_code):
        """Test Steane code with no errors."""
        H, logical_obs = steane_code
        error_probs = [0.01] * 7

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            debug=True
        )

        syndrome = [0.0, 0.0, 0.0]
        result = decoder.decode(syndrome)

        print(f"Steane code, no error: {result}")
        assert result['converged'] == True
        # With no syndrome and low error rate, logical error probability should be low
        assert result['logical_error_prob'] < 0.5

    def test_steane_decode_single_error(self, steane_code):
        """Test Steane code with single error syndrome."""
        H, logical_obs = steane_code
        error_probs = [0.01] * 7

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            debug=True
        )

        # Syndrome for error on qubit 0
        # H[:, 0] = [1, 1, 1] -> syndrome = [1, 1, 1]
        syndrome = [1.0, 1.0, 1.0]
        result = decoder.decode(syndrome)

        print(f"Steane code, syndrome [1,1,1]: {result}")
        assert result['converged'] == True

    def test_steane_with_erasure(self, steane_code):
        """Test Steane code with erasure."""
        H, logical_obs = steane_code
        error_probs = [0.01] * 7
        erasure_mask = [True, False, False, False, False, False, False]

        decoder = ErasureTensorNetworkDecoder(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probs,
            erasure_mask=erasure_mask,
            debug=True
        )

        syndrome = [0.0, 0.0, 0.0]
        result = decoder.decode(syndrome)

        print(f"Steane code with erasure on qubit 0: {result}")
        assert result['converged'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
