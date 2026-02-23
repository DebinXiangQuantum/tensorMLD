# ============================================================================ #
# Simple Decoders for Comparison                                               #
# ============================================================================ #
"""
Decoders for comparison with the erasure tensor network MLD decoder.

Provides wrappers over `ldpc` and (optionally) `cudaq_qec` decoders.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
import numpy.typing as npt

# Import ldpc decoders
try:
    from ldpc import BpDecoder, BpOsdDecoder, BpLsdDecoder, BeliefFindDecoder
    LDPC_AVAILABLE = True
except ImportError:
    LDPC_AVAILABLE = False
    print("[WARNING] ldpc package not available. Using fallback decoders.")

try:
    import cudaq_qec as qec
    CUDAQ_QEC_AVAILABLE = True
except ImportError:
    qec = None
    CUDAQ_QEC_AVAILABLE = False


class LookupTableDecoder:
    """
    Simple lookup table decoder that finds the most likely error for each syndrome.

    This is an exhaustive decoder that pre-computes all possible syndromes.
    Only suitable for small codes.
    """

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
    ):
        self.H = H.astype(np.int32)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

        # Build lookup table
        self._build_lookup_table()

    def _build_lookup_table(self):
        """Build lookup table mapping syndromes to most likely error patterns."""
        self.lookup_table = {}

        # Effective error probabilities (p=0.5 for erased qubits)
        effective_probs = self.error_probs.copy()
        effective_probs[self.erasure_mask] = 0.5

        # For small codes, enumerate all possible errors
        if self.num_errors <= 15:  # Only for small codes
            for error_int in range(2 ** self.num_errors):
                error = np.array([(error_int >> i) & 1 for i in range(self.num_errors)], dtype=np.int32)
                syndrome = tuple((self.H @ error) % 2)

                # Compute probability of this error
                log_prob = 0.0
                for i, (e, p) in enumerate(zip(error, effective_probs)):
                    if e == 1:
                        log_prob += np.log(p + 1e-10)
                    else:
                        log_prob += np.log(1 - p + 1e-10)

                # Compute logical value
                logical = int((self.logical_obs @ error) % 2)

                if syndrome not in self.lookup_table:
                    self.lookup_table[syndrome] = {'log_prob': log_prob, 'logical': logical, 'error': error.copy()}
                elif log_prob > self.lookup_table[syndrome]['log_prob']:
                    self.lookup_table[syndrome] = {'log_prob': log_prob, 'logical': logical, 'error': error.copy()}

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        """Update erasure mask and rebuild lookup table."""
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)
        self._build_lookup_table()

    def decode(self, syndrome: List[float]) -> Dict:
        """Decode a syndrome."""
        # Convert soft syndrome to hard decision
        hard_syndrome = tuple(int(s > 0.5) for s in syndrome)

        if hard_syndrome in self.lookup_table:
            result = self.lookup_table[hard_syndrome]
            return {
                'logical_error_prob': float(result['logical']),
                'converged': True,
                'result': [float(result['logical'])]
            }
        else:
            # Unknown syndrome - return 0.5
            return {
                'logical_error_prob': 0.5,
                'converged': False,
                'result': [0.5]
            }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        """Decode a batch of syndromes."""
        return [self.decode(s.tolist()) for s in syndromes]


class GreedyDecoder:
    """
    Greedy decoder that flips the most likely error at each step.

    This is a simple heuristic decoder that iteratively finds the most
    likely single error that explains the remaining syndrome.
    """

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
    ):
        self.H = H.astype(np.int32)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        """Update erasure mask."""
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)

    def decode(self, syndrome: List[float]) -> Dict:
        """Decode using greedy algorithm."""
        # Convert to hard syndrome
        current_syndrome = np.array([int(s > 0.5) for s in syndrome], dtype=np.int32)

        # Effective error probabilities
        effective_probs = self.error_probs.copy()
        effective_probs[self.erasure_mask] = 0.5

        # Track which errors we've assigned
        error_pattern = np.zeros(self.num_errors, dtype=np.int32)

        # Greedy loop
        max_iterations = self.num_errors
        for _ in range(max_iterations):
            if np.sum(current_syndrome) == 0:
                break

            # Find the error that explains the most syndrome bits
            best_score = -1
            best_idx = -1

            for i in range(self.num_errors):
                # How many syndrome bits does this error explain?
                syndrome_contribution = self.H[:, i]
                explained = np.sum((current_syndrome == 1) & (syndrome_contribution == 1))
                created = np.sum((current_syndrome == 0) & (syndrome_contribution == 1))
                score = explained - created

                # Weight by error probability
                weighted_score = score * effective_probs[i]

                if weighted_score > best_score:
                    best_score = weighted_score
                    best_idx = i

            if best_idx >= 0 and best_score > 0:
                error_pattern[best_idx] = 1 - error_pattern[best_idx]
                current_syndrome = (current_syndrome + self.H[:, best_idx]) % 2
            else:
                break

        # Compute logical
        logical = int((self.logical_obs @ error_pattern) % 2)

        return {
            'logical_error_prob': float(logical),
            'converged': np.sum(current_syndrome) == 0,
            'result': [float(logical)]
        }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        """Decode a batch of syndromes."""
        return [self.decode(s.tolist()) for s in syndromes]


class BeliefPropagationDecoder:
    """
    Simple Belief Propagation decoder.

    Implements min-sum belief propagation for binary codes.
    """

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_iterations: int = 50,
    ):
        self.H = H.astype(np.int32)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape
        self.max_iterations = max_iterations

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

        # Build adjacency information
        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency lists for efficient message passing."""
        self.check_to_var = [[] for _ in range(self.num_checks)]
        self.var_to_check = [[] for _ in range(self.num_errors)]

        for c in range(self.num_checks):
            for v in range(self.num_errors):
                if self.H[c, v] == 1:
                    self.check_to_var[c].append(v)
                    self.var_to_check[v].append(c)

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        """Update erasure mask."""
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)

    def decode(self, syndrome: List[float]) -> Dict:
        """Decode using belief propagation."""
        # Convert to hard syndrome
        hard_syndrome = np.array([int(s > 0.5) for s in syndrome], dtype=np.int32)

        # Effective error probabilities
        effective_probs = self.error_probs.copy()
        effective_probs[self.erasure_mask] = 0.5

        # Initialize log-likelihood ratios
        # LLR = log(P(e=0) / P(e=1)) = log((1-p)/p)
        llr = np.zeros(self.num_errors)
        for i in range(self.num_errors):
            p = effective_probs[i]
            llr[i] = np.log((1 - p + 1e-10) / (p + 1e-10))

        # Messages from variables to checks
        var_to_check_msg = np.zeros((self.num_errors, self.num_checks))
        # Messages from checks to variables
        check_to_var_msg = np.zeros((self.num_checks, self.num_errors))

        # Initialize var_to_check messages
        for v in range(self.num_errors):
            for c in self.var_to_check[v]:
                var_to_check_msg[v, c] = llr[v]

        converged = False
        for iteration in range(self.max_iterations):
            # Check to variable messages
            for c in range(self.num_checks):
                for v in self.check_to_var[c]:
                    # Min-sum update
                    product_sign = 1 if hard_syndrome[c] == 0 else -1
                    min_abs = float('inf')

                    for v2 in self.check_to_var[c]:
                        if v2 != v:
                            msg = var_to_check_msg[v2, c]
                            product_sign *= np.sign(msg) if msg != 0 else 1
                            min_abs = min(min_abs, abs(msg))

                    if min_abs == float('inf'):
                        min_abs = 0

                    check_to_var_msg[c, v] = product_sign * min_abs

            # Variable to check messages
            old_decisions = (llr < 0).astype(np.int32)

            for v in range(self.num_errors):
                total = llr[v]
                for c in self.var_to_check[v]:
                    total += check_to_var_msg[c, v]

                for c in self.var_to_check[v]:
                    var_to_check_msg[v, c] = total - check_to_var_msg[c, v]

            # Check convergence
            decisions = np.zeros(self.num_errors, dtype=np.int32)
            for v in range(self.num_errors):
                total = llr[v]
                for c in self.var_to_check[v]:
                    total += check_to_var_msg[c, v]
                decisions[v] = 1 if total < 0 else 0

            # Check if syndrome is satisfied
            computed_syndrome = (self.H @ decisions) % 2
            if np.array_equal(computed_syndrome, hard_syndrome):
                converged = True
                break

        # Compute logical
        logical = int((self.logical_obs @ decisions) % 2)

        return {
            'logical_error_prob': float(logical),
            'converged': converged,
            'result': [float(logical)]
        }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        """Decode a batch of syndromes."""
        return [self.decode(s.tolist()) for s in syndromes]


def _effective_error_probs(
    error_probabilities: np.ndarray,
    erasure_mask: np.ndarray,
) -> np.ndarray:
    """Apply p=0.5 prior on erased positions."""
    effective = error_probabilities.copy()
    effective[erasure_mask] = 0.5
    return effective


def _hard_syndrome(syndrome: List[float]) -> np.ndarray:
    """Convert soft syndrome to hard binary syndrome."""
    return np.array([int(s > 0.5) for s in syndrome], dtype=np.uint8)


def _logical_from_error(logical_obs: np.ndarray, error_estimate: np.ndarray) -> int:
    """Compute logical bit from estimated physical error."""
    return int(np.asarray((logical_obs @ error_estimate) % 2).reshape(-1)[0])


class LdpcBpOsdDecoder:
    """BP+OSD decoder using the `ldpc` package."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 50,
        osd_order: int = 7,
    ):
        if not LDPC_AVAILABLE:
            raise ImportError("ldpc package is required for LdpcBpOsdDecoder")

        self.H = H.astype(np.uint8)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities, dtype=np.float64)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

        self.max_bp_iter = max_bp_iter
        self.osd_order = osd_order
        self._build_decoder()

    def _build_decoder(self):
        effective_probs = _effective_error_probs(self.error_probs, self.erasure_mask)
        self.decoder = BpOsdDecoder(
            self.H,
            error_channel=effective_probs.tolist(),
            max_iter=self.max_bp_iter,
            bp_method="minimum_sum",
            osd_method="OSD_CS",
            osd_order=self.osd_order,
            input_vector_type="syndrome",
        )

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)
        self._build_decoder()

    def decode(self, syndrome: List[float]) -> Dict:
        hard = _hard_syndrome(syndrome)
        try:
            estimate = np.asarray(self.decoder.decode(hard), dtype=np.uint8).reshape(-1)
            logical = _logical_from_error(self.logical_obs, estimate)
            computed = (self.H @ estimate) % 2
            converged = np.array_equal(computed, hard)
            return {
                'logical_error_prob': float(logical),
                'converged': bool(converged),
                'result': [float(logical)]
            }
        except Exception as exc:
            return {
                'logical_error_prob': 0.5,
                'converged': False,
                'result': [0.5],
                'error': str(exc),
            }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        return [self.decode(s.tolist()) for s in syndromes]


class LdpcMinSumBpDecoder:
    """Min-Sum BP decoder using `ldpc.BpDecoder`."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 50,
        schedule: str = "parallel",
    ):
        if not LDPC_AVAILABLE:
            raise ImportError("ldpc package is required for LdpcMinSumBpDecoder")

        self.H = H.astype(np.uint8)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities, dtype=np.float64)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

        self.max_bp_iter = max_bp_iter
        self.schedule = schedule
        self._build_decoder()

    def _build_decoder(self):
        effective_probs = _effective_error_probs(self.error_probs, self.erasure_mask)
        self.decoder = BpDecoder(
            self.H,
            error_channel=effective_probs.tolist(),
            max_iter=self.max_bp_iter,
            bp_method="minimum_sum",
            schedule=self.schedule,
            input_vector_type="syndrome",
        )

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)
        self._build_decoder()

    def decode(self, syndrome: List[float]) -> Dict:
        hard = _hard_syndrome(syndrome)
        try:
            estimate = np.asarray(self.decoder.decode(hard), dtype=np.uint8).reshape(-1)
            logical = _logical_from_error(self.logical_obs, estimate)
            computed = (self.H @ estimate) % 2
            converged = np.array_equal(computed, hard)
            return {
                'logical_error_prob': float(logical),
                'converged': bool(converged),
                'result': [float(logical)]
            }
        except Exception as exc:
            return {
                'logical_error_prob': 0.5,
                'converged': False,
                'result': [0.5],
                'error': str(exc),
            }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        return [self.decode(s.tolist()) for s in syndromes]


class LdpcSequentialRelayBpDecoder:
    """Approximate Sequential Relay BP using multiple serial Min-Sum legs."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 50,
        num_legs: int = 4,
        random_seed: int = 42,
    ):
        if not LDPC_AVAILABLE:
            raise ImportError(
                "ldpc package is required for LdpcSequentialRelayBpDecoder")

        self.H = H.astype(np.uint8)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities, dtype=np.float64)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

        self.max_bp_iter = max_bp_iter
        self.num_legs = max(1, int(num_legs))
        self.random_seed = int(random_seed)
        self._build_decoders()

    def _build_decoders(self):
        effective_probs = _effective_error_probs(self.error_probs, self.erasure_mask)
        self.decoders = []
        for leg in range(self.num_legs):
            decoder = BpDecoder(
                self.H,
                error_channel=effective_probs.tolist(),
                max_iter=self.max_bp_iter,
                bp_method="minimum_sum",
                schedule="serial",
                input_vector_type="syndrome",
            )
            # Best effort randomization of serial order across relay legs.
            if hasattr(decoder, "random_schedule_seed"):
                decoder.random_schedule_seed = self.random_seed + leg
            if hasattr(decoder, "random_serial_schedule"):
                decoder.random_serial_schedule = True
            self.decoders.append(decoder)

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)
        self._build_decoders()

    def decode(self, syndrome: List[float]) -> Dict:
        hard = _hard_syndrome(syndrome)
        best: Optional[Dict[str, Any]] = None
        for decoder in self.decoders:
            try:
                estimate = np.asarray(decoder.decode(hard), dtype=np.uint8).reshape(-1)
                logical = _logical_from_error(self.logical_obs, estimate)
                computed = (self.H @ estimate) % 2
                residual = int(np.sum(computed != hard))
                converged = bool(residual == 0)
                candidate = {
                    'logical_error_prob': float(logical),
                    'converged': converged,
                    'result': [float(logical)],
                    '_residual': residual,
                }
                if best is None or candidate['_residual'] < best['_residual']:
                    best = candidate
                if converged:
                    break
            except Exception:
                continue

        if best is None:
            return {
                'logical_error_prob': 0.5,
                'converged': False,
                'result': [0.5],
                'error': 'all relay legs failed',
            }

        best.pop('_residual', None)
        return best

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        return [self.decode(s.tolist()) for s in syndromes]


class LdpcBpLsdDecoder:
    """BP+LSD decoder using the `ldpc` package."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 50,
        lsd_order: int = 0,
    ):
        if not LDPC_AVAILABLE:
            raise ImportError("ldpc package is required for LdpcBpLsdDecoder")

        self.H = H.astype(np.uint8)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities, dtype=np.float64)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

        self.max_bp_iter = max_bp_iter
        self.lsd_order = lsd_order
        self._build_decoder()

    def _build_decoder(self):
        effective_probs = _effective_error_probs(self.error_probs, self.erasure_mask)
        self.decoder = BpLsdDecoder(
            self.H,
            error_channel=effective_probs.tolist(),
            max_iter=self.max_bp_iter,
            bp_method="minimum_sum",
            lsd_method="LSD_CS",
            lsd_order=self.lsd_order,
            input_vector_type="syndrome",
        )

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)
        self._build_decoder()

    def decode(self, syndrome: List[float]) -> Dict:
        hard = _hard_syndrome(syndrome)
        try:
            estimate = np.asarray(self.decoder.decode(hard), dtype=np.uint8).reshape(-1)
            logical = _logical_from_error(self.logical_obs, estimate)
            computed = (self.H @ estimate) % 2
            converged = np.array_equal(computed, hard)
            return {
                'logical_error_prob': float(logical),
                'converged': bool(converged),
                'result': [float(logical)]
            }
        except Exception as exc:
            return {
                'logical_error_prob': 0.5,
                'converged': False,
                'result': [0.5],
                'error': str(exc),
            }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        return [self.decode(s.tolist()) for s in syndromes]


class LdpcBeliefFindDecoder:
    """Belief-Find decoder using the `ldpc` package."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 50,
    ):
        if not LDPC_AVAILABLE:
            raise ImportError("ldpc package is required for LdpcBeliefFindDecoder")

        self.H = H.astype(np.uint8)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities, dtype=np.float64)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

        self.max_bp_iter = max_bp_iter
        self._build_decoder()

    def _build_decoder(self):
        effective_probs = _effective_error_probs(self.error_probs, self.erasure_mask)
        self.decoder = BeliefFindDecoder(
            self.H,
            error_channel=effective_probs.tolist(),
            max_iter=self.max_bp_iter,
            bp_method="minimum_sum",
            uf_method="inversion",
            input_vector_type="syndrome",
        )

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)
        self._build_decoder()

    def decode(self, syndrome: List[float]) -> Dict:
        hard = _hard_syndrome(syndrome)
        try:
            estimate = np.asarray(self.decoder.decode(hard), dtype=np.uint8).reshape(-1)
            logical = _logical_from_error(self.logical_obs, estimate)
            computed = (self.H @ estimate) % 2
            converged = np.array_equal(computed, hard)
            return {
                'logical_error_prob': float(logical),
                'converged': bool(converged),
                'result': [float(logical)]
            }
        except Exception as exc:
            return {
                'logical_error_prob': 0.5,
                'converged': False,
                'result': [0.5],
                'error': str(exc),
            }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        return [self.decode(s.tolist()) for s in syndromes]


class CudaqNvQldpcBaseDecoder:
    """Base wrapper for CUDA-Q `nv-qldpc-decoder` under erasure-aware priors."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 50,
        **decoder_kwargs,
    ):
        if not CUDAQ_QEC_AVAILABLE or qec is None:
            raise ImportError(
                "cudaq_qec is required for CudaqNvQldpcBaseDecoder")

        self.H = H.astype(np.uint8)
        self.logical_obs = logical_obs.astype(np.int32)
        self.num_checks, self.num_errors = H.shape

        if isinstance(error_probabilities, np.ndarray):
            self.error_probs = error_probabilities.copy()
        else:
            self.error_probs = np.array(error_probabilities, dtype=np.float64)

        if erasure_mask is None:
            self.erasure_mask = np.zeros(self.num_errors, dtype=bool)
        else:
            self.erasure_mask = np.array(erasure_mask, dtype=bool)

        self.max_bp_iter = max_bp_iter
        self.decoder_kwargs = dict(decoder_kwargs)
        self._build_decoder()

    def _build_decoder(self):
        effective_probs = _effective_error_probs(self.error_probs, self.erasure_mask)
        kwargs = dict(self.decoder_kwargs)
        kwargs.setdefault("max_iterations", self.max_bp_iter)
        kwargs["error_rate_vec"] = effective_probs.astype(np.float64)
        self.decoder = qec.get_decoder("nv-qldpc-decoder", self.H, **kwargs)

    def update_erasure_mask(self, new_erasure_mask: Union[List[bool], np.ndarray]):
        self.erasure_mask = np.array(new_erasure_mask, dtype=bool)
        self._build_decoder()

    def decode(self, syndrome: List[float]) -> Dict:
        hard = _hard_syndrome(syndrome)
        try:
            output = self.decoder.decode(hard.tolist())
            if hasattr(output, "result"):
                estimate = np.asarray(output.result, dtype=np.uint8).reshape(-1)
                converged = bool(getattr(output, "converged", True))
            else:
                estimate = np.asarray(output, dtype=np.uint8).reshape(-1)
                converged = True

            logical = _logical_from_error(self.logical_obs, estimate)
            return {
                'logical_error_prob': float(logical),
                'converged': bool(converged),
                'result': [float(logical)]
            }
        except Exception as exc:
            return {
                'logical_error_prob': 0.5,
                'converged': False,
                'result': [0.5],
                'error': str(exc),
            }

    def decode_batch(self, syndromes: npt.NDArray[Any]) -> List[Dict]:
        return [self.decode(s.tolist()) for s in syndromes]


class CudaqBpOsdDecoder(CudaqNvQldpcBaseDecoder):
    """BP+OSD using CUDA-Q `nv-qldpc-decoder`."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 50,
        osd_order: int = 7,
    ):
        super().__init__(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probabilities,
            erasure_mask=erasure_mask,
            max_bp_iter=max_bp_iter,
            bp_method=0,
            use_sparsity=True,
            use_osd=True,
            osd_order=osd_order,
            osd_method=1,
        )


class CudaqMinSumBpDecoder(CudaqNvQldpcBaseDecoder):
    """Min-Sum BP using CUDA-Q `nv-qldpc-decoder`."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 50,
        scale_factor: float = 1.0,
    ):
        super().__init__(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probabilities,
            erasure_mask=erasure_mask,
            max_bp_iter=max_bp_iter,
            bp_method=1,
            use_sparsity=True,
            use_osd=False,
            scale_factor=scale_factor,
        )


class CudaqSequentialRelayBpDecoder(CudaqNvQldpcBaseDecoder):
    """Sequential Relay BP using CUDA-Q `nv-qldpc-decoder`."""

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        error_probabilities: Union[List[float], np.ndarray],
        erasure_mask: Optional[Union[List[bool], np.ndarray]] = None,
        max_bp_iter: int = 60,
        pre_iter: int = 5,
        num_sets: int = 4,
        gamma0: float = 0.3,
        gamma_dist: Optional[List[float]] = None,
        bp_seed: int = 42,
    ):
        if gamma_dist is None:
            gamma_dist = [0.1, 0.6]
        srelay_config = {
            "pre_iter": int(pre_iter),
            "num_sets": int(num_sets),
            "stopping_criterion": "FirstConv",
        }
        super().__init__(
            H=H,
            logical_obs=logical_obs,
            error_probabilities=error_probabilities,
            erasure_mask=erasure_mask,
            max_bp_iter=max_bp_iter,
            bp_method=3,
            composition=1,
            use_sparsity=True,
            use_osd=False,
            gamma0=float(gamma0),
            gamma_dist=list(gamma_dist),
            srelay_config=srelay_config,
            bp_seed=int(bp_seed),
        )
