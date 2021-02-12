# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Look for the supported GPU languages.
include( vecmem-check-language )
vecmem_check_language( CUDA )
vecmem_check_language( HIP )

# Set up the project's options.
include( CMakeDependentOption )

# Flag specifying whether CUDA support should be built.
cmake_dependent_option( VECMEM_BUILD_CUDA_LIBRARY
   "Build the vecmem::cuda library" ON
   "CMAKE_CUDA_COMPILER" OFF )

# Flag specifying whether HIP support should be built.
cmake_dependent_option( VECMEM_BUILD_HIP_LIBRARY
   "Build the vecmem::hip library" ON
   "CMAKE_HIP_COMPILER" OFF )

# Flag specifying whether SYCL support should be built.
option( VECMEM_BUILD_SYCL_LIBRARY "Build the vecmem::sycl library" OFF )
