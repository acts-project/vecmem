# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CUDAToolkit requires CMake 3.17.
cmake_minimum_required( VERSION 3.17 )

# Enable CUDA as a language.
enable_language( CUDA )

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-cuda )

# External dependency/dependencies.
find_package( CUDAToolkit REQUIRED )

add_library(
   vecmem_testing_cuda_main
   STATIC
   test_cuda_main.cpp
   "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_error_handling.hpp"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_error_handling.cpp"
)

target_link_libraries(
   vecmem_testing_cuda_main
   PUBLIC
   CUDA::cudart
   GTest::gtest
   vecmem::core
)

# Test all of the CUDA library's features.
vecmem_add_test( cuda
   "test_cuda_memory_resources.cpp"
   "test_cuda_containers.cpp" "test_cuda_containers_kernels.cuh"
   "test_cuda_containers_kernels.cu"
   "test_cuda_edm_copy.cpp"
   "test_cuda_edm.cpp"
   "test_cuda_edm_kernels.hpp"
   "test_cuda_edm_kernels.cu"
   "test_cuda_jagged_vector_view.cpp"
   "test_cuda_jagged_vector_view_kernels.cu"
   "test_cuda_jagged_vector_view_kernels.cuh"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_error_handling.hpp"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_error_handling.cpp"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_wrappers.hpp"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_wrappers.cpp"
   LINK_LIBRARIES CUDA::cudart vecmem::core vecmem::cuda vecmem_testing_cuda_main
                  vecmem_testing_common )

# Set up a separate test that would ensure a C++17 standard. But only with
# CUDA 11+.
if( "${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "11.0" )
   vecmem_add_test( cuda_cxx17
      "test_cuda_memory_resources.cu"
      "test_cuda_containers.cu" "test_cuda_containers_kernels.cuh"
      "test_cuda_containers_kernels.cu"
      "test_cuda_edm_copy.cu"
      "test_cuda_edm.cu"
      "test_cuda_edm_kernels.hpp"
      "test_cuda_edm_kernels.cu"
      "test_cuda_jagged_vector_view.cu"
      "test_cuda_jagged_vector_view_kernels.cu"
      "test_cuda_jagged_vector_view_kernels.cuh"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_error_handling.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_error_handling.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_wrappers.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/src/utils/cuda_wrappers.cpp"
      LINK_LIBRARIES vecmem::core vecmem::cuda vecmem_testing_cuda_main
                     vecmem_testing_common )
   set_target_properties( vecmem_test_cuda_cxx17 PROPERTIES
      CUDA_STANDARD 17 )
endif()
