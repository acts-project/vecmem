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
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (host) C++ standard to use" )

# Turn on a number of warnings for the "known compilers".
if( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" ) OR
    ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" ) )

   # Basic flags for all build modes.
   foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
      vecmem_add_flag( CMAKE_CXX_FLAGS_${mode} "-Wall" )
      vecmem_add_flag( CMAKE_CXX_FLAGS_${mode} "-Wextra" )
   endforeach()

   # More rigorous tests for the Debug builds.
   vecmem_add_flag( CMAKE_CXX_FLAGS_DEBUG "-Werror" )
   vecmem_add_flag( CMAKE_CXX_FLAGS_DEBUG "-pedantic" )
   vecmem_add_flag( CMAKE_CXX_FLAGS_DEBUG "-fsanitize=undefined" )
endif()
