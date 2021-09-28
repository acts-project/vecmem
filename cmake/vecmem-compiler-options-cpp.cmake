# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( vecmem-functions )

# Set up the used C++ standard(s).
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (host) C++ standard to use" )

# Do not export symbols by default.
set( CMAKE_CXX_VISIBILITY_PRESET "hidden" CACHE STRING
   "C++ symbol visibility setting" )

# Turn on the correct setting for the __cplusplus macro with MSVC.
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
   vecmem_add_flag( CMAKE_CXX_FLAGS "/Zc:__cplusplus" )
endif()

# Turn on a number of warnings for the "known compilers".
if( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" ) OR
    ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" ) )

   # Basic flags for all build modes.
   foreach( mode RELEASE RELWITHDEBINFO MINSIZEREL DEBUG )
      vecmem_add_flag( CMAKE_CXX_FLAGS_${mode} "-Wall" )
      vecmem_add_flag( CMAKE_CXX_FLAGS_${mode} "-Wextra" )
      vecmem_add_flag( CMAKE_CXX_FLAGS_${mode} "-Wshadow" )
      vecmem_add_flag( CMAKE_CXX_FLAGS_${mode} "-Wunused-local-typedefs" )
   endforeach()

   # More rigorous tests for the Debug builds.
   vecmem_add_flag( CMAKE_CXX_FLAGS_DEBUG "-Werror" )
   vecmem_add_flag( CMAKE_CXX_FLAGS_DEBUG "-pedantic" )

elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )

   # Basic flags for all build modes.
   string( REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS}" )
   vecmem_add_flag( CMAKE_CXX_FLAGS "/W4" )

   # More rigorous tests for the Debug builds.
   vecmem_add_flag( CMAKE_CXX_FLAGS_DEBUG "/WX" )

endif()
