# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Include the helper function(s).
include( "${CMAKE_CURRENT_LIST_DIR}/vecmem-functions.cmake" )

# Set up the used C++ standard(s).
set( CMAKE_CUDA_STANDARD 14 CACHE STRING "The (CUDA) C++ standard to use" )

# Set the architecture to build code for.
set( CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING
   "CUDA architectures to build device code for" )

# Make CUDA generate debug symbols for the device code as well in a debug
# build.
vecmem_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-G" )
