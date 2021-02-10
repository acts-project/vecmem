# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# Project include(s).
include(
   "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/vecmem-compiler-options-cuda.cmake" )

# Tests for the CUDA library's custom memory management.
vecmem_add_test( cuda_allocators test_cuda_allocators.cpp
   LINK_LIBRARIES vecmem::core vecmem::cuda )

# Tests for the core library's device vector types, using the CUDA backend.
vecmem_add_test( cuda_containers test_cuda_containers.cu
   LINK_LIBRARIES vecmem::core vecmem::cuda )
