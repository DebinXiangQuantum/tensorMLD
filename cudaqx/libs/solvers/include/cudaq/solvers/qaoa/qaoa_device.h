/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"

namespace cudaq::solvers {

/// @brief QAOA quantum kernel implementation for quantum approximate
/// optimization
/// @details This kernel implements the Quantum Approximate Optimization
/// Algorithm (QAOA) circuit structure, which alternates between problem and
/// mixing Hamiltonians for a specified number of layers. The circuit begins
/// with an equal superposition state and applies parameterized evolution under
/// both Hamiltonians.
///
/// The circuit structure is:
/// 1. Initialize all qubits in superposition with Hadamard gates
/// 2. For each layer:
///    - Apply problem Hamiltonian evolution with gamma parameters
///    - Apply mixing Hamiltonian evolution with beta parameters
///
/// @param numQubits Number of qubits in the QAOA circuit
/// @param numLayers Number of QAOA layers (p-value) to apply
/// @param gamma_beta Vector of alternating gamma/beta variational parameters
/// @param problemHCoeffs Coefficients for each term in the problem Hamiltonian
/// @param problemH Pauli string operators representing the problem Hamiltonian
/// terms
/// @param referenceHCoeffs Coefficients for each term in the mixing Hamiltonian
/// @param referenceH Pauli string operators representing the mixing Hamiltonian
/// terms
/// @param full_parameterization if true, use a parameter for every term in both
/// problem and reference Hamltonians.
/// @param counterdiabatic if true, add ry rotations to every qubit after
/// reference Hamiltonian.
///
/// @see qaoa_result For the structure containing optimization results
/// @see exp_pauli For the primitive implementing parameterized Pauli evolution
__qpu__ void qaoa_kernel(std::size_t numQubits, std::size_t numLayers,
                         const std::vector<double> &gamma_beta,
                         const std::vector<double> &problemHCoeffs,
                         const std::vector<cudaq::pauli_word> &problemH,
                         const std::vector<double> &referenceHCoeffs,
                         const std::vector<cudaq::pauli_word> &referenceH,
                         bool full_parameterization, bool counterdiabatic);
} // namespace cudaq::solvers
