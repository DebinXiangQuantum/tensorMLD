"""Cycle-level Ising machine simulator and hardware metrics.

Simulates the dynamics of a BRIM (Bistable Resistively-coupled Ising Machine)
for LDPC/QLDPC decoding. The simulator discretizes the continuous-time ODE:

    dv_i/dt = (J/C) * [bias_i + coupling_i]

where:
    bias_i    = (4 * R_i / alpha)           -- channel reliability pull
    coupling_i = sum_{j: i in check_j} sigma_{j\\i}  -- parity coupling
    sigma_{j\\i} = (-1)^{s_j} * prod_{k in check_j, k != i} sigma_k

Annealing schedule: coupling strength J ramps from 0 to J_max following
an exponential schedule, while spin-fix perturbations decay exponentially.

Reference: Section 3.3 (Architecture) and Section 4 (Evaluation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SimulationResult:
    """Result of an Ising machine simulation run."""
    spins: np.ndarray                     # final spin configuration (+1/-1)
    errors: np.ndarray                    # decoded error pattern (0/1)
    num_cycles: int                       # total simulation cycles
    converged: bool                       # all parity checks satisfied?
    parity_violations: int                # number of unsatisfied checks
    energy_trace: list[float] = field(default_factory=list)
    violation_trace: list[int] = field(default_factory=list)


class IsingMachineSim:
    """Cycle-level simulator for the BRIM Ising machine.

    Models the discrete-time dynamics of an analog Ising machine where:
    - Spins are capacitor voltage polarities
    - Couplings are resistive connections through XNOR gates (parity units)
    - Bias currents encode channel reliability (LLR)
    - Annealing ramps coupling strength and decays perturbation frequency

    Args:
        H: Parity check matrix, shape (num_checks, num_errors).
        alpha: Relative weight between bias and parity coupling.
        J_max: Maximum coupling strength (normalized).
        tau_anneal: Annealing time constant (controls ramp speed).
        perturb_rate_init: Initial spin-fix perturbation probability.
        perturb_decay: Perturbation decay rate.
        v_clip: Voltage clipping (supply rail magnitude).
        dt: Discretization time step (normalized).
    """

    def __init__(
        self,
        H: np.ndarray,
        alpha: float = 2.0,
        J_max: float = 1.0,
        tau_anneal: float = 0.3,
        perturb_rate_init: float = 0.05,
        perturb_decay: float = 3.0,
        v_clip: float = 1.0,
        dt: float = 0.01,
    ):
        self.H = np.asarray(H, dtype=np.int8)
        self.m, self.n = self.H.shape
        self.alpha = alpha
        self.J_max = J_max
        self.tau_anneal = tau_anneal
        self.perturb_rate_init = perturb_rate_init
        self.perturb_decay = perturb_decay
        self.v_clip = v_clip
        self.dt = dt

        # Precompute sparse structure
        self.check_qubits: list[np.ndarray] = []
        for j in range(self.m):
            self.check_qubits.append(np.where(self.H[j] != 0)[0])

        self.qubit_checks: list[np.ndarray] = []
        for i in range(self.n):
            self.qubit_checks.append(np.where(self.H[:, i] != 0)[0])

    def simulate(
        self,
        syndrome: np.ndarray,
        llr: np.ndarray,
        num_cycles: int = 500,
        seed: Optional[int] = None,
        trace_interval: int = 50,
    ) -> SimulationResult:
        """Run one complete annealing simulation.

        Args:
            syndrome: binary syndrome, shape (m,).
            llr: log-likelihood ratios, shape (n,).
            num_cycles: number of discrete time steps.
            seed: random seed.
            trace_interval: record energy every N cycles.

        Returns:
            SimulationResult with decoded spins and diagnostics.
        """
        rng = np.random.default_rng(seed)
        syndrome = np.asarray(syndrome, dtype=np.int8)
        llr = np.asarray(llr, dtype=np.float64)
        syn_signs = np.where(syndrome == 0, 1.0, -1.0)

        # Initialize capacitor voltages from LLR (hard decision + small noise)
        v = np.clip(llr / (np.max(np.abs(llr)) + 1e-12), -0.5, 0.5)
        v += rng.normal(0, 0.1, self.n)

        # Bias currents (proportional to channel reliability)
        bias = (4.0 / self.alpha) * llr

        energy_trace: list[float] = []
        violation_trace: list[int] = []

        for cycle in range(num_cycles):
            frac = cycle / max(1, num_cycles - 1)

            # --- Annealing: ramp coupling strength ---
            J = self.J_max * (1.0 - math.exp(-frac / self.tau_anneal))

            # --- Quantize spins from voltages ---
            spins = np.sign(v)
            spins[spins == 0] = 1.0  # break ties

            # --- Compute full check parities ---
            check_parities = np.ones(self.m)
            for j in range(self.m):
                check_parities[j] = np.prod(spins[self.check_qubits[j]])

            # --- Update voltages (parallel update, all spins simultaneously) ---
            dv = np.zeros(self.n)
            for i in range(self.n):
                coupling = 0.0
                for j in self.qubit_checks[i]:
                    # sigma_{j\i} = (-1)^{s_j} * P_j / sigma_i
                    # This "backs out" qubit i's contribution from the check product
                    sigma_ji = syn_signs[j] * check_parities[j] * spins[i]
                    # NOTE: multiplying by spins[i] is equivalent to dividing,
                    # since spins[i] in {+1,-1} and spins[i]^2 = 1.
                    coupling += sigma_ji
                dv[i] = bias[i] + J * coupling

            v += self.dt * dv
            v = np.clip(v, -self.v_clip, self.v_clip)

            # --- Spin-fix perturbation (escape local minima) ---
            perturb_rate = self.perturb_rate_init * math.exp(
                -self.perturb_decay * frac
            )
            perturb_mask = rng.random(self.n) < perturb_rate
            if np.any(perturb_mask):
                # Clip perturbed spins to supply rails
                v[perturb_mask] = self.v_clip * rng.choice(
                    [-1.0, 1.0], size=int(np.sum(perturb_mask))
                )

            # --- Trace ---
            if trace_interval > 0 and cycle % trace_interval == 0:
                e_bias = -np.dot(llr, spins)
                violations = 0
                e_parity = 0.0
                for j in range(self.m):
                    term = 1.0 - syn_signs[j] * check_parities[j]
                    e_parity += term
                    if term > 0.5:
                        violations += 1
                energy_trace.append(e_bias + (self.alpha / 2.0) * e_parity)
                violation_trace.append(violations)

        # --- Final readout ---
        final_spins = np.sign(v)
        final_spins[final_spins == 0] = 1.0
        errors = ((1.0 - final_spins) / 2.0).astype(np.int8)

        # Count final parity violations
        final_violations = 0
        for j in range(self.m):
            check_val = int(np.sum(errors[self.check_qubits[j]])) % 2
            if check_val != int(syndrome[j]):
                final_violations += 1

        return SimulationResult(
            spins=final_spins,
            errors=errors,
            num_cycles=num_cycles,
            converged=(final_violations == 0),
            parity_violations=final_violations,
            energy_trace=energy_trace,
            violation_trace=violation_trace,
        )


def ising_machine_metrics(
    H: np.ndarray,
    annealing_time_ns: float = 320.0,
    cycle_time_ns: float = 4.0,
    bp_max_iter: int = 50,
) -> dict:
    """Compute hardware metrics for an Ising machine decoding a given code.

    Estimates resource usage and latency based on the BRIM architecture
    described in the paper (Section 3.3 and Section 4).

    Args:
        H: Parity check matrix, shape (num_checks, num_errors).
        annealing_time_ns: Annealing time in nanoseconds.
        cycle_time_ns: Hardware clock cycle time in nanoseconds.
        bp_max_iter: BP initialization iterations (each iter = 2 cycles).

    Returns:
        Dict with hardware metrics including cycle counts and latency.
    """
    H = np.asarray(H)
    m, n = H.shape

    row_weights = np.sum(H != 0, axis=1)
    col_weights = np.sum(H != 0, axis=0)
    nnz = int(np.sum(H != 0))

    max_rw = int(np.max(row_weights))
    max_cw = int(np.max(col_weights))
    avg_rw = float(np.mean(row_weights))

    xnor_depth = int(math.ceil(math.log2(max_rw))) if max_rw > 1 else 0

    # Throughput: block_size / annealing_time
    throughput_gbps = n / annealing_time_ns  # bits / ns = Gbps

    # Standard QUBO auxiliary variable counts (for comparison)
    # Unary encoding: each check j needs floor(rw_j/2) auxiliary vars
    aux_unary = int(np.sum(row_weights // 2))
    # Binary encoding: each check j needs ceil(log2(floor(rw_j/2)+1)) aux vars
    aux_binary = 0
    for rw in row_weights:
        L_max = int(rw) // 2
        if L_max > 0:
            aux_binary += int(math.ceil(math.log2(L_max + 1)))

    # Coupler counts
    # Standard unary: O((n + aux_unary)^2) upper bound, actual is sparser
    # We estimate based on the quadratization structure
    n_unary = n + aux_unary
    n_binary = n + aux_binary
    # For modified QUBO: couplers = nnz(H) (no auxiliary variables)
    couplers_modified = nnz

    # Cycle counts and hardware latency
    # Annealing cycles: annealing_time / cycle_time
    annealing_cycles = int(math.ceil(annealing_time_ns / cycle_time_ns))
    # BP initialization: each iteration has forward + backward pass = 2 cycles
    bp_init_cycles = bp_max_iter * 2
    # Total hardware cycles = BP init + Ising annealing
    total_cycles = bp_init_cycles + annealing_cycles
    # Hardware latency = total_cycles * cycle_time
    hardware_latency_ns = total_cycles * cycle_time_ns

    return {
        "num_spins": n,
        "num_checks": m,
        "num_couplers_modified_qubo": couplers_modified,
        "nnz_H": nnz,
        "max_row_weight": max_rw,
        "max_col_weight": max_cw,
        "avg_row_weight": round(avg_rw, 2),
        "xnor_tree_depth": xnor_depth,
        "annealing_time_ns": annealing_time_ns,
        "cycle_time_ns": cycle_time_ns,
        "annealing_cycles": annealing_cycles,
        "bp_init_cycles": bp_init_cycles,
        "total_cycles": total_cycles,
        "hardware_latency_ns": hardware_latency_ns,
        "estimated_throughput_gbps": round(throughput_gbps, 4),
        "parity_units": m,
        "total_spins_standard_unary": n_unary,
        "total_spins_standard_binary": n_binary,
        "aux_vars_unary": aux_unary,
        "aux_vars_binary": aux_binary,
        "spin_overhead_unary_pct": round(100.0 * aux_unary / n, 1),
        "spin_overhead_binary_pct": round(100.0 * aux_binary / n, 1),
    }
