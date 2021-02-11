# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Find the HIP compiler.
find_program( CMAKE_HIP_COMPILER NAMES hipcc
   PATHS "${HIP_ROOT_DIR}"
         ENV ROCM_PATH
         ENV HIP_PATH
         "/opt/rocm"
         "/opt/rocm/hip"
   PATH_SUFFIXES "bin"
   DOC "HIP compiler to use" )
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
set( CMAKE_INCLUDE_FLAG_SEP_HIP " ${CMAKE_INCLUDE_FLAG_HIP}" )

# Set up the linker used for components holding HIP source code.
set( CMAKE_HIP_HOST_LINKER "${CMAKE_CXX_COMPILER}" )

# Decide whether to generate AMD or NVidia code using HIP.
set( CMAKE_HIP_PLATFORM_DEFAULT "hcc" )
if( NOT "$ENV{HIP_PLATFORM}" STREQUAL "" )
   set( CMAKE_HIP_PLATFORM_DEFAULT "$ENV{HIP_PLATFORM}" )
endif()
set( CMAKE_HIP_PLATFORM "${CMAKE_HIP_PLATFORM_DEFAULT}" CACHE STRING
   "Platform to build the HIP code for" )
set_property( CACHE CMAKE_HIP_PLATFORM
   PROPERTY STRINGS "hcc" "nvcc" )

# Decide how to do the build for the AMD (hcc) and NVidia (nvcc) backends.
if( "${CMAKE_HIP_PLATFORM}" STREQUAL "hcc" )
   if( CMAKE_HIP_VERSION VERSION_LESS "3.7" )
      set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "-x c++" )
   else()
      set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "" )
   endif()
   set( CMAKE_HIP_IMPLICIT_LINK_LIBRARIES "${HIP_amdhip64_LIBRARY}" )
   set( CMAKE_HIP_COMPILE_OPTIONS_PIC "${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )
elseif( "${CMAKE_HIP_PLATFORM}" STREQUAL "nvcc" )
   set( CMAKE_HIP_COMPILE_SOURCE_TYPE_FLAG "-x cu" )
   find_package( CUDAToolkit QUIET REQUIRED )
   set( CMAKE_HIP_IMPLICIT_LINK_LIBRARIES "${CUDA_cudart_LIBRARY}" )
   set( CMAKE_HIP_COMPILE_OPTIONS_PIC "${CMAKE_CUDA_COMPILE_OPTIONS_PIC}" )
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
string( APPEND CMAKE_HIP_FLAGS
   "${CMAKE_HIP${CMAKE_HIP_STANDARD}_STANDARD_COMPILE_OPTION}" )

# Configure variables set in this file for fast reload later on.
configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMakeHIPCompiler.cmake.in
   ${CMAKE_PLATFORM_INFO_DIR}/CMakeHIPCompiler.cmake )
