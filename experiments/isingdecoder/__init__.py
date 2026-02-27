"""Ising machine-based LDPC/QLDPC decoder using modified QUBO formulation.

Implements the decoding approach that converts QEC syndrome decoding into
a modified QUBO (Quadratic Unconstrained Binary Optimization) problem,
solvable by Ising machines or simulated annealing.

Key insight (from paper Section 3.2): by using spin representation
(sigma_i = 1 - 2*e_i), XOR becomes multiplication, eliminating the need
for auxiliary binary variables. This keeps the state space at 2^n.
"""

from .qubo_decoder import IsingDecoder
from .ising_machine_sim import IsingMachineSim, ising_machine_metrics

__all__ = ["IsingDecoder", "IsingMachineSim", "ising_machine_metrics"]
