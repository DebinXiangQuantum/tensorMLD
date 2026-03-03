"""Modified QUBO decoder for LDPC/QLDPC codes.

Converts syndrome decoding to a modified QUBO problem and solves via
simulated annealing (SA). The modified QUBO formulation avoids auxiliary
spins by exploiting the spin representation where XOR -> multiplication.

Energy function (spin representation, sigma_i in {+1, -1}):

    E = -sum_i R_i * sigma_i
        + (alpha/2) * sum_j (1 - (-1)^{s_j} * prod_{k in check_j} sigma_k)

Key improvements over naive SA:
  1. Adaptive alpha: auto-scaled so parity can compete with LLR bias
  2. GF(2) initialization: find a valid error pattern first, then optimize
  3. Stabilizer row-flip moves: syndrome-preserving exploration of the coset
  4. Optional BP seeding: use BP output as starting point
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np


@dataclass
class DecoderResult:
    """Decode result compatible with cudaq_qec.DecoderResult."""
    converged: bool = True
    result: list[float] = field(default_factory=list)


@dataclass
class DecodeTimingInfo:
    """Timing breakdown for a single decode call."""
    total_ms: float = 0.0
    sa_ms: float = 0.0
    num_reads: int = 0
    num_steps: int = 0
    actual_sa_steps: int = 0       # steps actually used (with early stopping)
    bp_init_iters: int = 0         # actual BP initialization iterations
    best_energy: float = float("inf")
    parity_violations: int = 0


# ---------------------------------------------------------------------------
# GF(2) utilities
# ---------------------------------------------------------------------------
def _gf2_solve(H: np.ndarray, s: np.ndarray) -> Optional[np.ndarray]:
    """Find a particular solution e to H @ e = s (mod 2).

    Uses Gaussian elimination over GF(2).

    Args:
        H: (m, n) binary matrix.
        s: (m,) binary vector.

    Returns:
        (n,) binary solution vector, or None if infeasible.
    """
    m, n = H.shape
    # Augmented matrix [H | s]
    aug = np.hstack([H.astype(np.int8), s.reshape(-1, 1).astype(np.int8)])

    pivot_cols = []
    row = 0
    for col in range(n):
        # Find pivot in this column from current row downward
        found = -1
        for r in range(row, m):
            if aug[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        # Swap rows
        if found != row:
            aug[[row, found]] = aug[[found, row]]
        pivot_cols.append(col)
        # Eliminate other rows
        for r in range(m):
            if r != row and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[row]) % 2
        row += 1

    # Check consistency: if any row has [0 0 ... 0 | 1], infeasible
    for r in range(row, m):
        if aug[r, n] == 1:
            return None

    # Back-substitute: set free variables to 0
    e = np.zeros(n, dtype=np.int8)
    for idx, col in enumerate(pivot_cols):
        e[col] = aug[idx, n]
    return e


def _gf2_kernel_basis(H: np.ndarray) -> np.ndarray:
    """Compute a basis for the kernel of H over GF(2).

    Returns rows that span ker(H) = {x : Hx = 0 mod 2}.
    These correspond to stabilizer generators and can be used
    for syndrome-preserving moves.
    """
    m, n = H.shape
    # Row-reduce H over GF(2)
    aug = H.astype(np.int8).copy()

    pivot_cols: list[tuple[int, int]] = []  # (col, row) pairs
    row = 0
    for col in range(n):
        found = -1
        for r in range(row, m):
            if aug[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        if found != row:
            aug[[row, found]] = aug[[found, row]]
        pivot_cols.append((col, row))
        for r in range(m):
            if r != row and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[row]) % 2
        row += 1

    # Free columns (not pivot columns) -> kernel dimension
    pivot_set = {col for col, _ in pivot_cols}
    free_cols = [c for c in range(n) if c not in pivot_set]
    if not free_cols:
        return np.zeros((0, n), dtype=np.int8)

    # For each free column fc, build a kernel vector:
    #   vec[fc] = 1, vec[pc] = aug[r, fc] for each pivot (pc, r)
    basis = []
    for fc in free_cols:
        vec = np.zeros(n, dtype=np.int8)
        vec[fc] = 1
        for pc, r in pivot_cols:
            vec[pc] = aug[r, fc]
        basis.append(vec)

    return np.array(basis, dtype=np.int8)


# ---------------------------------------------------------------------------
# Optional BP initialization
# ---------------------------------------------------------------------------
def _try_bp_init(
    H: np.ndarray, syndrome: np.ndarray, noise_p: float,
    max_iter: int = 50,
) -> tuple[Optional[np.ndarray], int]:
    """Try to use BP decoder for initialization.

    Returns:
        (error_pattern or None, actual_iterations_used).
    """
    try:
        from ldpc import BpDecoder
        H_int = H.astype(np.uint8)
        channel_probs = np.full(H.shape[1], noise_p)
        bp = BpDecoder(H_int, error_rate=noise_p,
                       channel_probs=channel_probs,
                       max_iter=max_iter, bp_method="ms")
        correction = bp.decode(syndrome.astype(np.uint8))
        # ldpc BpDecoder exposes .iter for actual iterations used
        bp_iters = getattr(bp, "iter", max_iter)
        if bp_iters is None or bp_iters <= 0:
            bp_iters = max_iter
        return correction.astype(np.int8), int(bp_iters)
    except Exception:
        return None, 0


class IsingDecoder:
    """Ising machine-based LDPC/QLDPC decoder using modified QUBO.

    Solves the syndrome decoding problem by formulating it as an Ising
    optimization problem and solving via simulated annealing with:
      - Adaptive alpha (auto-scaled to LLR magnitude)
      - GF(2) initialization (syndrome-valid starting point)
      - Stabilizer row-flip moves (syndrome-preserving coset search)
      - Optional BP seeding

    Args:
        H: Parity check matrix, shape (num_checks, num_errors).
        logical_obs: Logical operators, shape (K, num_errors).
        noise_model: Per-qubit error probabilities, length num_errors.
        alpha: Parity constraint weight. "auto" scales by LLR * col_weight.
        num_reads: Number of independent SA runs per decode.
        num_steps_per_qubit: SA steps per qubit per read.
        T_init: Initial SA temperature.
        T_final: Final SA temperature.
        schedule: Annealing schedule: "linear" or "exponential".
        use_bp_init: Use BP decoder output as one of the initial reads.
        use_stabilizer_moves: Mix in syndrome-preserving row-flip moves.
        stabilizer_move_prob: Fraction of SA steps that use stabilizer moves.
        seed: Random seed for reproducibility (None = random).
    """

    def __init__(
        self,
        H: np.ndarray,
        logical_obs: np.ndarray,
        noise_model: Union[list[float], np.ndarray, float],
        alpha: Union[float, str] = "auto",
        num_reads: int = 16,
        num_steps_per_qubit: int = 50,
        T_init: float = 2.0,
        T_final: float = 1e-4,
        schedule: str = "linear",
        use_bp_init: bool = True,
        use_stabilizer_moves: bool = True,
        stabilizer_move_prob: float = 0.3,
        convergence_patience: int = 0,
        bp_max_iter: int = 50,
        seed: Optional[int] = None,
    ):
        self.H = np.asarray(H, dtype=np.int8)
        self.logical_obs = np.asarray(logical_obs, dtype=np.int8)
        self.m, self.n = self.H.shape
        self.K = self.logical_obs.shape[0]

        # Noise model -> per-qubit LLR
        if np.isscalar(noise_model):
            p = np.full(self.n, float(noise_model))
        else:
            p = np.asarray(noise_model, dtype=np.float64)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        self.noise_p = p
        self.llr = np.log((1.0 - p) / p)

        # Adaptive alpha: scale so parity penalty can overcome LLR bias.
        # A single-spin flip costs 2*R in bias. To compensate, each check
        # it participates in contributes alpha. Need alpha * col_weight > 2*R.
        if alpha == "auto":
            col_weights = np.sum(self.H != 0, axis=0).astype(float)
            mean_cw = max(float(np.mean(col_weights)), 1.0)
            mean_R = float(np.mean(np.abs(self.llr)))
            self.alpha = max(4.0 * mean_R / mean_cw, 2.0)
        else:
            self.alpha = float(alpha)

        # SA parameters
        self.num_reads = num_reads
        self.num_steps = num_steps_per_qubit * self.n
        self.T_init = T_init
        self.T_final = T_final
        self.schedule = schedule
        self.use_bp_init = use_bp_init
        self.use_stabilizer_moves = use_stabilizer_moves
        self.stabilizer_move_prob = stabilizer_move_prob
        # Early stopping: 0 = disabled, >0 = stop if no energy improvement
        # for this many consecutive steps
        self.convergence_patience = convergence_patience
        self.bp_max_iter = bp_max_iter
        self.rng = np.random.default_rng(seed)

        # Precompute neighborhood structure (sparse H)
        self.check_qubits: list[np.ndarray] = []
        for j in range(self.m):
            self.check_qubits.append(np.where(self.H[j] != 0)[0])

        self.qubit_checks: list[np.ndarray] = []
        for i in range(self.n):
            self.qubit_checks.append(np.where(self.H[:, i] != 0)[0])

        # Precompute stabilizer basis for syndrome-preserving moves
        self._stab_basis: Optional[np.ndarray] = None
        if use_stabilizer_moves:
            self._stab_basis = _gf2_kernel_basis(self.H)
            if self._stab_basis.shape[0] == 0:
                self._stab_basis = None

        self.last_timing: Optional[DecodeTimingInfo] = None

    def _errors_to_spins(self, errors: np.ndarray) -> np.ndarray:
        """Convert binary error pattern to spin representation."""
        return 1.0 - 2.0 * errors.astype(np.float64)

    def _spins_to_errors(self, spins: np.ndarray) -> np.ndarray:
        """Convert spin representation to binary error pattern."""
        return ((1.0 - spins) / 2.0).astype(np.int8)

    def _compute_syndrome(self, errors: np.ndarray) -> np.ndarray:
        """Compute syndrome from error pattern."""
        return (errors @ self.H.T) % 2

    def _energy(
        self, spins: np.ndarray, syn_signs: np.ndarray
    ) -> tuple[float, int]:
        """Compute modified QUBO energy and count parity violations."""
        bias = -np.dot(self.llr, spins)
        parity_energy = 0.0
        violations = 0
        for j in range(self.m):
            P_j = np.prod(spins[self.check_qubits[j]])
            term = 1.0 - syn_signs[j] * P_j
            parity_energy += term
            if term > 0.5:
                violations += 1
        return bias + (self.alpha / 2.0) * parity_energy, violations

    def _init_candidates(
        self, syndrome: np.ndarray
    ) -> tuple[list[np.ndarray], int]:
        """Generate diverse initial error patterns.

        Returns:
            (list of binary error patterns, actual BP iterations used).
        """
        candidates: list[np.ndarray] = []
        bp_iters_used = 0

        # 1. GF(2) exact solution (guarantees syndrome satisfaction)
        gf2_sol = _gf2_solve(self.H, syndrome)
        if gf2_sol is not None:
            candidates.append(gf2_sol.copy())

        # 2. BP initialization (if available and enabled)
        if self.use_bp_init:
            bp_sol, bp_iters_used = _try_bp_init(
                self.H, syndrome, float(self.noise_p[0]),
                max_iter=self.bp_max_iter,
            )
            if bp_sol is not None:
                candidates.append(bp_sol.copy())

        # 3. Hard decision (all zeros — no errors)
        candidates.append(np.zeros(self.n, dtype=np.int8))

        # 4. Random perturbations of GF(2) solution
        base = gf2_sol if gf2_sol is not None else np.zeros(self.n, dtype=np.int8)
        if self._stab_basis is not None and len(self._stab_basis) > 0:
            for _ in range(max(0, self.num_reads - len(candidates))):
                perturbed = base.copy()
                # Randomly add some stabilizer rows
                n_flips = self.rng.integers(1, max(2, len(self._stab_basis) // 2 + 1))
                rows_to_add = self.rng.choice(
                    len(self._stab_basis), size=min(n_flips, len(self._stab_basis)),
                    replace=False)
                for r in rows_to_add:
                    perturbed = (perturbed + self._stab_basis[r]) % 2
                candidates.append(perturbed)
        else:
            # Fallback: random perturbations
            for _ in range(max(0, self.num_reads - len(candidates))):
                perturbed = base.copy()
                flip_prob = 0.05 + 0.15 * self.rng.random()
                flip_mask = self.rng.random(self.n) < flip_prob
                perturbed[flip_mask] = 1 - perturbed[flip_mask]
                candidates.append(perturbed)

        return candidates[:self.num_reads], bp_iters_used

    def _simulated_annealing(
        self, syndrome: np.ndarray
    ) -> tuple[np.ndarray, float, int, int, int]:
        """Run SA to minimize modified QUBO energy.

        Uses:
          - GF(2) / BP / stabilizer-perturbed initialization
          - Mix of single-spin flips and stabilizer row-flip moves
          - Optional convergence-based early stopping

        Returns:
            (best_spins, best_energy, best_violations,
             total_actual_sa_steps, bp_init_iters)
        """
        syn_signs = np.where(syndrome == 0, 1.0, -1.0)
        candidates, bp_iters_used = self._init_candidates(syndrome)

        best_spins = None
        best_energy = float("inf")
        best_violations = self.m
        total_actual_steps = 0

        for read_idx in range(self.num_reads):
            # Initialize from candidate (or random if we ran out)
            if read_idx < len(candidates):
                errors = candidates[read_idx].copy()
            else:
                errors = np.zeros(self.n, dtype=np.int8)
                flip_mask = self.rng.random(self.n) < 0.1
                errors[flip_mask] = 1

            spins = self._errors_to_spins(errors)

            # Maintain check products incrementally
            check_products = np.empty(self.m)
            for j in range(self.m):
                check_products[j] = np.prod(spins[self.check_qubits[j]])

            # --- SA loop with optional early stopping ---
            read_best_energy = float("inf")
            no_improve_count = 0
            actual_steps_this_read = 0
            for step in range(self.num_steps):
                actual_steps_this_read = step + 1
                frac = step / max(1, self.num_steps - 1)
                if self.schedule == "exponential":
                    T = self.T_init * (self.T_final / self.T_init) ** frac
                else:
                    T = self.T_init * (1.0 - frac) + self.T_final * frac

                # Decide move type
                use_stab = (
                    self.use_stabilizer_moves
                    and self._stab_basis is not None
                    and self.rng.random() < self.stabilizer_move_prob
                )

                if use_stab:
                    # --- Stabilizer row-flip move (syndrome-preserving) ---
                    row_idx = self.rng.integers(len(self._stab_basis))
                    stab_row = self._stab_basis[row_idx]
                    flip_qubits = np.where(stab_row != 0)[0]

                    if len(flip_qubits) == 0:
                        continue

                    # Compute delta_E for flipping all qubits in the stabilizer
                    # Bias change: sum over flipped qubits of 2 * R_i * sigma_i
                    delta_E = 0.0
                    for i in flip_qubits:
                        delta_E += 2.0 * self.llr[i] * spins[i]

                    # Parity change: for each check, compute new product after
                    # flipping all involved qubits. Since it's a stabilizer,
                    # syndrome is preserved, so parity violations stay the same
                    # in expectation. But the check products change.
                    # Actually: flipping a kernel vector preserves He mod 2,
                    # so ALL check products that matter stay consistent.
                    # But in the spin energy, the check product P_j changes
                    # by (-1)^{number of flipped qubits in check j}.
                    affected_checks = set()
                    for i in flip_qubits:
                        for j in self.qubit_checks[i]:
                            affected_checks.add(j)

                    parity_delta = 0.0
                    for j in affected_checks:
                        n_flips_in_check = int(np.sum(stab_row[self.check_qubits[j]]))
                        if n_flips_in_check % 2 == 1:
                            # P_j flips sign
                            parity_delta += self.alpha * syn_signs[j] * check_products[j]
                        # If even number of flips in check, P_j unchanged
                    delta_E += parity_delta

                    # Metropolis
                    if delta_E < 0.0 or self.rng.random() < np.exp(
                        -delta_E / max(T, 1e-30)
                    ):
                        for i in flip_qubits:
                            spins[i] *= -1.0
                        for j in affected_checks:
                            n_flips_in_check = int(np.sum(stab_row[self.check_qubits[j]]))
                            if n_flips_in_check % 2 == 1:
                                check_products[j] *= -1.0
                else:
                    # --- Single-spin flip move ---
                    i = self.rng.integers(self.n)
                    delta_E = 2.0 * self.llr[i] * spins[i]
                    for j in self.qubit_checks[i]:
                        delta_E += self.alpha * syn_signs[j] * check_products[j]

                    if delta_E < 0.0 or self.rng.random() < np.exp(
                        -delta_E / max(T, 1e-30)
                    ):
                        spins[i] *= -1.0
                        for j in self.qubit_checks[i]:
                            check_products[j] *= -1.0

                # --- Convergence check (periodically, every n steps) ---
                if self.convergence_patience > 0 and step % self.n == 0 and step > 0:
                    cur_energy, _ = self._energy(spins, syn_signs)
                    if cur_energy < read_best_energy - 1e-8:
                        read_best_energy = cur_energy
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    if no_improve_count >= self.convergence_patience:
                        break  # early stop this read

            total_actual_steps += actual_steps_this_read

            # Evaluate final energy
            energy, violations = self._energy(spins, syn_signs)
            if energy < best_energy:
                best_energy = energy
                best_spins = spins.copy()
                best_violations = violations

        return best_spins, best_energy, best_violations, total_actual_steps, bp_iters_used

    def decode(self, syndrome: Union[list[float], np.ndarray]) -> DecoderResult:
        """Decode a single syndrome."""
        t0 = time.perf_counter()
        syndrome = np.asarray(syndrome, dtype=np.int8)

        best_spins, best_energy, violations, actual_sa_steps, bp_init_iters = (
            self._simulated_annealing(syndrome)
        )
        errors = self._spins_to_errors(best_spins)
        logical_vals = (errors @ self.logical_obs.T) % 2

        elapsed = (time.perf_counter() - t0) * 1e3
        self.last_timing = DecodeTimingInfo(
            total_ms=elapsed, sa_ms=elapsed,
            num_reads=self.num_reads, num_steps=self.num_steps,
            actual_sa_steps=actual_sa_steps,
            bp_init_iters=bp_init_iters,
            best_energy=best_energy, parity_violations=violations,
        )

        res = DecoderResult()
        res.converged = violations == 0
        res.result = [float(v) for v in logical_vals]
        return res

    def decode_batch(
        self, syndromes: np.ndarray, verbose: bool = False,
    ) -> list[DecoderResult]:
        """Decode a batch of syndromes."""
        results = []
        n_shots = syndromes.shape[0]
        for i in range(n_shots):
            res = self.decode(syndromes[i])
            results.append(res)
            if verbose and (i + 1) % max(1, n_shots // 10) == 0:
                print(
                    f"  [{i+1}/{n_shots}] converged={res.converged}, "
                    f"violations={self.last_timing.parity_violations}")
        return results

    def benchmark_decode_latency(
        self, syndrome: Union[list[float], np.ndarray],
        warmup: int = 5, repeats: int = 20,
    ) -> dict[str, Any]:
        """Benchmark single-syndrome decode latency."""
        syndrome = np.asarray(syndrome, dtype=np.int8)
        for _ in range(warmup):
            self.decode(syndrome)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.decode(syndrome)
            times.append((time.perf_counter() - t0) * 1e3)
        return {
            "method": "modified_qubo_sa",
            "alpha": self.alpha,
            "avg_ms": float(np.mean(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "std_ms": float(np.std(times)),
            "num_reads": self.num_reads,
            "num_steps": self.num_steps,
        }
