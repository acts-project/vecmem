# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# The SYCL language code only works with the Ninja, and the different kinds
# of Makefile generators.
if( NOT ( ( "${CMAKE_GENERATOR}" MATCHES "Make" ) OR
          ( "${CMAKE_GENERATOR}" MATCHES "Ninja" ) ) )
   message( FATAL_ERROR "SYCL language not currently supported by "
      "\"${CMAKE_GENERATOR}\" generator" )
endif()

# Use the SYCLCXX environment variable preferably as the SYCL compiler.
if( NOT "$ENV{SYCLCXX}" STREQUAL "" )

   # Interpret the contents of SYCLCXX.
   get_filename_component( CMAKE_SYCL_COMPILER_INIT "$ENV{SYCLCXX}"
      PROGRAM PROGRAM_ARGS CMAKE_SYCL_FLAGS_INIT )
   if( NOT EXISTS ${CMAKE_SYCL_COMPILER_INIT} )
      message( FATAL_ERROR
         "Could not find compiler set in environment variable SYCLCXX:\n$ENV{SYCLCXX}.\n${CMAKE_SYCL_COMPILER_INIT}")
   endif()

   # Determine the version of the SYCL compiler.
   execute_process( COMMAND "${CMAKE_SYCL_COMPILER_INIT}" "--version"
      OUTPUT_VARIABLE _syclVersionOutput
      RESULT_VARIABLE _syclVersionResult )
   if( ${_syclVersionResult} EQUAL 0 )
      string( REPLACE "\n" ";" _syclVersionOutputList "${_syclVersionOutput}" )
      foreach( _line ${_syclVersionOutputList} )
         if( "${_line}" MATCHES "[a-zA-Z]+ version ([0-9\.]+)" )
            set( CMAKE_SYCL_COMPILER_VERSION "${CMAKE_MATCH_1}" )
            break()
         endif()
      endforeach()
      unset( _syclVersionOutputList )
   else()
      message( WARNING "Failed to execute: ${CMAKE_SYCL_COMPILER_INIT} --version" )
      set( CMAKE_SYCL_COMPILER_VERSION "Unknown" )
   endif()
   unset( _syclVersionOutput )
   unset( _syclVersionResult )
else()
   # If not specified otherwise, try to use the C++ compiler.
   set( CMAKE_SYCL_COMPILER_INIT "${CMAKE_CXX_COMPILER}" )
   set( CMAKE_SYCL_FLAGS_INIT "-fsycl" )
   set( CMAKE_SYCL_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}" )
endif()

# Set up the compiler with cache variables.
set( CMAKE_SYCL_COMPILER "${CMAKE_SYCL_COMPILER_INIT}" CACHE FILEPATH
   "The SYCL compiler to use. Normally the same as the C++ compiler." )

# Set up what source/object file names to use.
set( CMAKE_SYCL_SOURCE_FILE_EXTENSIONS "sycl" )
set( CMAKE_SYCL_OUTPUT_EXTENSION ".o" )
set( CMAKE_SYCL_COMPILER_ENV_VAR "SYCLCXX" )

# Set how the compiler should pass include directories to the SYCL compiler.
set( CMAKE_INCLUDE_FLAG_SYCL "${CMAKE_INCLUDE_FLAG_CXX}" )
set( CMAKE_INCLUDE_SYSTEM_FLAG_SYCL "${CMAKE_INCLUDE_SYSTEM_FLAG_CXX}" )

# Set up the linker used for components holding SYCL source code.
set( CMAKE_SYCL_HOST_LINKER "${CMAKE_SYCL_COMPILER}" )

# Flag used for building position independent code. (The same as that of the
# C++ compiler.)
set( CMAKE_SYCL_COMPILE_OPTIONS_PIC "${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )

# Default (optimisation) flags. (Heavily based on the C++ compiler.)
set( CMAKE_SYCL_FLAGS_INIT
   "${CMAKE_SYCL_FLAGS_INIT} ${CMAKE_CXX_FLAGS_INIT} $ENV{SYCLFLAGS}" )
set( CMAKE_SYCL_FLAGS_DEBUG_INIT "${CMAKE_CXX_FLAGS_DEBUG_INIT}" )
set( CMAKE_SYCL_FLAGS_RELEASE_INIT "${CMAKE_CXX_FLAGS_RELEASE_INIT}" )
set( CMAKE_SYCL_FLAGS_RELWITHDEBINFO_INIT
   "${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT}" )

# Set up C++17 by default.
set( CMAKE_SYCL_STANDARD 17 CACHE STRING "C++ standard to use with SYCL" )
set_property( CACHE CMAKE_SYCL_STANDARD PROPERTY STRINGS 17 )

# Configure variables set in this file for fast reload later on.
configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMakeSYCLCompiler.cmake.in
   ${CMAKE_PLATFORM_INFO_DIR}/CMakeSYCLCompiler.cmake )
