# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
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
      vecmem_add_flag( CMAKE_HIP_FLAGS_${mode} "-Wshadow" )
      vecmem_add_flag( CMAKE_HIP_FLAGS_${mode} "-Wunused-local-typedefs" )
   endforeach()
endif()
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "amd" ) )
   foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
      vecmem_add_flag( CMAKE_HIP_FLAGS_${mode} "-pedantic" )
   endforeach()
endif()

# Generate debug symbols for the device code as well in a debug build.
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
   vecmem_add_flag( CMAKE_HIP_FLAGS_DEBUG "-G" )
endif()

# Fail on warnings, if asked for that behaviour.
if( VECMEM_FAIL_ON_WARNINGS )
   foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
      if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" ) OR
          ( "${CMAKE_HIP_PLATFORM}" STREQUAL "amd" ) )
         vecmem_add_flag( CMAKE_HIP_FLAGS_${mode} "-Werror" )
      elseif( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
              ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
         vecmem_add_flag( CMAKE_HIP_FLAGS_${mode} "-Werror all-warnings" )
      endif()
   endforeach()
endif()
