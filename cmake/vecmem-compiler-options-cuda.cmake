# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( vecmem-functions )

# Figure out the properties of CUDA being used.
find_package( CUDAToolkit REQUIRED )

# Set up the used C++ standard(s).
set( CMAKE_CUDA_STANDARD 14 CACHE STRING "The (CUDA) C++ standard to use" )

# Set the architecture to build code for.
set( CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING
   "CUDA architectures to build device code for" )

# Make CUDA generate debug symbols for the device code as well in a debug
# build.
if( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" )
   vecmem_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-G" )
endif()

# More rigorous tests for the Debug builds.
if( ( "${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "10.2" ) AND
    ( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" ) )
   vecmem_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-Werror all-warnings" )
elseif( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang" )
   vecmem_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-Werror" )
endif()
