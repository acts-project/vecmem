# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( vecmem-functions )

# Set up the used C++ standard(s).
set( CMAKE_HIP_STANDARD 14 CACHE STRING "The (HIP) C++ standard to use" )

# Basic flags for all build modes.
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "amd" ) )
   foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
      vecmem_add_flag( CMAKE_HIP_FLAGS_${mode} "-Wall" )
      vecmem_add_flag( CMAKE_HIP_FLAGS_${mode} "-Wextra" )
   endforeach()
endif()

# Generate debug symbols for the device code as well in a debug build.
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
   vecmem_add_flag( CMAKE_HIP_FLAGS_DEBUG "-G" )
endif()

# More rigorous tests for the Debug builds.
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "amd" ) )
   vecmem_add_flag( CMAKE_HIP_FLAGS_DEBUG "-Werror" )
   vecmem_add_flag( CMAKE_HIP_FLAGS_DEBUG "-pedantic" )
elseif( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
        ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
   vecmem_add_flag( CMAKE_HIP_FLAGS_DEBUG "-Werror all-warnings" )
endif()
