# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable HIP as a language.
enable_language( HIP )

# Project include(s).
include( vecmem-compiler-options-hip )

# Tests for the HIP library's custom memory management.
vecmem_add_test( hip_allocators test_hip_allocators.cpp
   LINK_LIBRARIES vecmem::core vecmem::hip )

# Tests for the core library's device vector types, using the HIP backend.
vecmem_add_test( hip_containers test_hip_containers.cpp
   test_hip_containers_kernels.hpp test_hip_containers_kernels.hip
   LINK_LIBRARIES vecmem::core vecmem::hip )
