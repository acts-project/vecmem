# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# Project include(s).
include( vecmem-compiler-options-cuda )

# Test all of the CUDA library's features.
vecmem_add_test( cuda
   "test_cuda_memory_resources.cpp"
   "test_cuda_containers.cpp" "test_cuda_containers_kernels.cuh"
   "test_cuda_containers_kernels.cu"
   LINK_LIBRARIES vecmem::core vecmem::cuda GTest::gtest_main
                  vecmem_testing_common )
