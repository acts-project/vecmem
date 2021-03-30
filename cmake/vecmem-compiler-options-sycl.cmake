# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Include the helper function(s).
include( vecmem-functions )
include( vecmem-options )

# Set up the used C++ standard(s).
set( CMAKE_SYCL_STANDARD 17 CACHE STRING "The (SYCL) C++ standard to use" )

# Basic flags for all build modes.
foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
   vecmem_add_flag( CMAKE_SYCL_FLAGS_${mode} "-Wall" )
   vecmem_add_flag( CMAKE_SYCL_FLAGS_${mode} "-Wextra" )
   vecmem_add_flag( CMAKE_SYCL_FLAGS_${mode} "-Wno-unknown-cuda-version" )
   vecmem_add_flag( CMAKE_SYCL_FLAGS_${mode}
      "-DVECMEM_DEBUG_MSG_LVL=${VECMEM_DEBUG_MSG_LVL}" )
endforeach()

# More rigorous tests for the Debug builds.
vecmem_add_flag( CMAKE_SYCL_FLAGS_DEBUG "-Werror" )
vecmem_add_flag( CMAKE_SYCL_FLAGS_DEBUG "-pedantic" )
