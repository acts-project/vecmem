# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Use the HIPCXX environment variable preferably as the HIP compiler.
if( NOT "$ENV{HIPCXX}" STREQUAL "" )
   # Interpret the contents of HIPCXX.
   get_filename_component( CMAKE_HIP_COMPILER_INIT $ENV{HIPCXX}
      PROGRAM PROGRAM_ARGS CMAKE_HIP_FLAGS_INIT )
   if( NOT EXISTS ${CMAKE_HIP_COMPILER_INIT} )
      message( FATAL_ERROR
         "Could not find compiler set in environment variable HIPCXX:\n$ENV{HIPCXX}.\n${CMAKE_HIP_COMPILER_INIT}")
   endif()
else()
   # Find the HIP compiler.
   find_program( CMAKE_HIP_COMPILER_INIT NAMES hipcc
      PATHS "${HIP_ROOT_DIR}"
            ENV ROCM_PATH
            ENV HIP_PATH
            "/opt/rocm"
            "/opt/rocm/hip"
      PATH_SUFFIXES "bin" )
   set( CMAKE_HIP_FLAGS_INIT "" )
endif()

# Set up the compiler as a cache variable.
set( CMAKE_HIP_COMPILER "${CMAKE_HIP_COMPILER_INIT}" CACHE FILEPATH
   "The HIP compiler to use" )
mark_as_advanced( CMAKE_HIP_COMPILER )

# If found, figure out the version of ROCm/HIP that we found.
if( CMAKE_HIP_COMPILER )
   execute_process( COMMAND ${CMAKE_HIP_COMPILER} --version
      OUTPUT_VARIABLE _hipVersionOutput
      RESULT_VARIABLE _hipVersionResult )
   if( ${_hipVersionResult} EQUAL 0 )
      string( REPLACE "\n" ";" _hipVersionOutputList "${_hipVersionOutput}" )
      foreach( _line ${_hipVersionOutputList} )
         if( "${_line}" MATCHES "HIP version: (.*)" )
            set( CMAKE_HIP_VERSION "${CMAKE_MATCH_1}" )
            break()
         endif()
      endforeach()
   else()
      message( WARNING "Failed to execute: ${CMAKE_HIP_COMPILER} --version" )
      set( CMAKE_HIP_VERSION "Unknown" )
   endif()
   unset( _hipVersionOutput )
   unset( _hipVersionResult )
endif()

# Look for the ROCm/HIP header(s).
find_path( HIP_INCLUDE_DIR
   NAMES "hip/hip_runtime.h"
         "hip/hip_runtime_api.h"
   PATHS "${HIP_ROOT_DIR}"
         ENV ROCM_PATH
         ENV HIP_PATH
         "/opt/rocm"
         "/opt/rocm/hip"
   PATH_SUFFIXES "include"
   DOC "ROCm/HIP include directory" )
mark_as_advanced( HIP_INCLUDE_DIR )

# Find the amdhip64 shared library.
find_library( HIP_amdhip64_LIBRARY NAMES amdhip64
   PATHS "${HIP_ROOT_DIR}"
         ENV ROCM_PATH
         ENV HIP_PATH
         "/opt/rocm"
         "/opt/rocm/hip"
   PATH_SUFFIXES "lib" "lib64"
   DOC "HIP library to use" )
mark_as_advanced( HIP_amdhip64_LIBRARY )

# Set up what source/object file names to use.
set( CMAKE_HIP_SOURCE_FILE_EXTENSIONS "hip" )
set( CMAKE_HIP_OUTPUT_EXTENSION ".o" )
set( CMAKE_HIP_COMPILER_ENV_VAR "HIPCXX" )

# Set how the compiler should pass include directories to the HIP compiler.
set( CMAKE_INCLUDE_FLAG_HIP "${CMAKE_INCLUDE_FLAG_CXX}" )
set( CMAKE_INCLUDE_SYSTEM_FLAG_HIP "${CMAKE_INCLUDE_SYSTEM_FLAG_CXX}" )

# Set up the linker used for components holding HIP source code.
set( CMAKE_HIP_HOST_LINKER "${CMAKE_HIP_COMPILER}" )

# Decide whether to generate AMD or NVidia code using HIP.
set( CMAKE_HIP_PLATFORM_DEFAULT "hcc" )
if( NOT "$ENV{HIP_PLATFORM}" STREQUAL "" )
   set( CMAKE_HIP_PLATFORM_DEFAULT "$ENV{HIP_PLATFORM}" )
endif()
set( CMAKE_HIP_PLATFORM "${CMAKE_HIP_PLATFORM_DEFAULT}" CACHE STRING
   "Platform to build the HIP code for" )
set_property( CACHE CMAKE_HIP_PLATFORM
   PROPERTY STRINGS "hcc" "nvcc" "amd" "nvidia" )

# Turn on CUDA support if we use nvcc.
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
   enable_language( CUDA )
endif()

# Decide how to do the build for the AMD (hcc) and NVidia (nvcc) backends.
set( CMAKE_HIP_FLAGS_INIT "${CMAKE_INCLUDE_SYSTEM_FLAG_HIP}${HIP_INCLUDE_DIR}" )
if( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" ) OR
    ( "${CMAKE_HIP_PLATFORM}" STREQUAL "amd" ) )
   if( CMAKE_HIP_VERSION VERSION_LESS "3.7" )
      set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "-x c++" )
   else()
      set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "" )
   endif()
   set( CMAKE_HIP_IMPLICIT_LINK_LIBRARIES "${HIP_amdhip64_LIBRARY}" )
   set( CMAKE_HIP_COMPILE_OPTIONS_PIC "${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )
   set( CMAKE_HIP_FLAGS_DEBUG_INIT "${CMAKE_CXX_FLAGS_DEBUG_INIT}" )
   set( CMAKE_HIP_FLAGS_RELEASE_INIT "${CMAKE_CXX_FLAGS_RELEASE_INIT}" )
   set( CMAKE_HIP_FLAGS_RELWITHDEBINFO_INIT
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT}" )
elseif( ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" ) OR
        ( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvidia" ) )
   set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "-x cu" )
   find_package( CUDAToolkit QUIET REQUIRED )
   set( CMAKE_HIP_IMPLICIT_LINK_LIBRARIES "${CUDA_cudart_LIBRARY}" )
   set( CMAKE_HIP_COMPILE_OPTIONS_PIC "${CMAKE_CUDA_COMPILE_OPTIONS_PIC}" )
   set( CMAKE_HIP_FLAGS_DEBUG_INIT "${CMAKE_CUDA_FLAGS_DEBUG_INIT}" )
   set( CMAKE_HIP_FLAGS_RELEASE_INIT "${CMAKE_CUDA_FLAGS_RELEASE_INIT}" )
   set( CMAKE_HIP_FLAGS_RELWITHDEBINFO_INIT
      "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT}" )
else()
   message( FATAL_ERROR
      "\nInvalid setting for CMAKE_HIP_PLATFORM (\"${CMAKE_HIP_PLATFORM}\").\n"
      "Use either \"hcc\" or \"nvcc\"!\n" )
endif()

# Tell CMake what compiler standards it may use with HIP.
set( CMAKE_HIP11_STANDARD_COMPILE_OPTION "-std=c++11" )
set( CMAKE_HIP11_EXTENSION_COMPILE_OPTION "-std=c++11" )

set( CMAKE_HIP14_STANDARD_COMPILE_OPTION "-std=c++14" )
set( CMAKE_HIP14_EXTENSION_COMPILE_OPTION "-std=c++14" )

set( CMAKE_HIP17_STANDARD_COMPILE_OPTION "-std=c++17" )
set( CMAKE_HIP17_EXTENSION_COMPILE_OPTION "-std=c++17" )

# Set up C++14 by default, as C++17 is not supported by the NVidia backend.
# However let the user choose C++17 if they build for the AMD backend. They'll
# get a clear-enough error message if they chose the wrong setting anyway.
set( CMAKE_HIP_STANDARD 14 CACHE STRING "C++ standard to use with HIP" )
set_property( CACHE CMAKE_HIP_STANDARD PROPERTY STRINGS 11 14 17 )

# Configure variables set in this file for fast reload later on.
configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMakeHIPCompiler.cmake.in
   ${CMAKE_PLATFORM_INFO_DIR}/CMakeHIPCompiler.cmake )
