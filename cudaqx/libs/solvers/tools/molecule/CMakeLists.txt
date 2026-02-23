# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

file(COPY cudaq-pyscf.py 
  DESTINATION ${CMAKE_BINARY_DIR}/bin
  FILE_PERMISSIONS
    OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

file(RENAME "${CMAKE_BINARY_DIR}/bin/cudaq-pyscf.py"
            "${CMAKE_BINARY_DIR}/bin/cudaq-pyscf")

# Install
# ==============================================================================

install(PROGRAMS cudaq-pyscf.py
  COMPONENT solvers-tools
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  RENAME cudaq-pyscf
)
