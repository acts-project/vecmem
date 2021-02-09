# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Look for CUDA.
include( CheckLanguage )
check_language( CUDA )

# Set up the project's options.
include( CMakeDependentOption )

cmake_dependent_option( VECMEM_BUILD_CUDA_LIBRARY
   "Build the vecmem::cuda library" ON
   "CMAKE_CUDA_COMPILER" OFF )

option( VECMEM_BUILD_SYCL_LIBRARY "Build the vecmem::sycl library" OFF )
