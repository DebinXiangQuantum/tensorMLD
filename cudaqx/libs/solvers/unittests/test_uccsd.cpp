/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq/solvers/operators.h"
#include "cudaq/solvers/stateprep/uccsd.h"
#include "cudaq/solvers/vqe.h"

#include "nvqpp/test_kernels.h"

TEST(SolversUCCSDTester, checkUCCSD) {

  cudaq::solvers::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                              {"H", {0., 0., .7474}}};
  auto molecule = cudaq::solvers::create_molecule(geometry, "sto-3g", 0, 0,
                                                  {.verbose = true});

  auto numElectrons = molecule.n_electrons;
  auto numQubits = molecule.n_orbitals * 2;

  // EXPECT_NEAR(molecule.fci_energy, -1.137, 1e-3);
  EXPECT_NEAR(molecule.energies["hf_energy"], -1.1163255644, 1e-3);
  EXPECT_EQ(numElectrons, 2);
  EXPECT_EQ(numQubits, 4);

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      cudaq::solvers::stateprep::get_uccsd_excitations(numElectrons, numQubits);
  EXPECT_TRUE(doublesAlpha.empty());
  EXPECT_TRUE(doublesBeta.empty());
  EXPECT_TRUE(singlesAlpha.size() == 1);
  EXPECT_EQ(singlesAlpha[0][0], 0);
  EXPECT_EQ(singlesAlpha[0][1], 2);
  EXPECT_EQ(singlesBeta[0][0], 1);
  EXPECT_EQ(singlesBeta[0][1], 3);
  EXPECT_EQ(doublesMixed[0][0], 0);
  EXPECT_EQ(doublesMixed[0][1], 1);
  EXPECT_EQ(doublesMixed[0][2], 3);
  EXPECT_EQ(doublesMixed[0][3], 2);
  EXPECT_TRUE(singlesBeta.size() == 1);
  EXPECT_TRUE(doublesMixed.size() == 1);

  auto numParams = cudaq::solvers::stateprep::get_num_uccsd_parameters(
      numElectrons, numQubits);
  EXPECT_EQ(numParams, 3);
  std::vector<double> init{-2., -2., -2.};
  auto optimizer = cudaq::optim::optimizer::get("cobyla");
  {
    auto result = cudaq::solvers::vqe(callUccsdStatePrep, molecule.hamiltonian,
                                      *optimizer, init, {{"verbose", true}});
    EXPECT_NEAR(result.energy, -1.137, 1e-3);
  }

  {
    auto result = cudaq::solvers::vqe(
        callUccsdStatePrepWithArgs, molecule.hamiltonian, *optimizer, init,
        [&](std::vector<double> x) { return std::make_tuple(x, 4, 2); },
        {{"verbose", true}});
    EXPECT_NEAR(result.energy, -1.137, 1e-3);
  }
}
