# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# The SYCL compiler, by definition, is the same as the C++ compiler.
set( CMAKE_SYCL_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
   "The SYCL compiler to use. Normally the same as the C++ compiler." )
set( CMAKE_SYCL_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}" )

# Set up what source/object file names to use.
set( CMAKE_SYCL_SOURCE_FILE_EXTENSIONS "sycl" )
set( CMAKE_SYCL_OUTPUT_EXTENSION ".o" )
set( CMAKE_SYCL_COMPILER_ENV_VAR "SYCLCXX" )

# Set how the compiler should pass include directories to the SYCL compiler.
set( CMAKE_INCLUDE_FLAG_SYCL "${CMAKE_INCLUDE_FLAG_CXX}" )
set( CMAKE_INCLUDE_SYSTEM_FLAG_SYCL "${CMAKE_INCLUDE_SYSTEM_FLAG_CXX}" )

# Set up the linker used for components holding SYCL source code.
set( CMAKE_SYCL_HOST_LINKER "${CMAKE_CXX_COMPILER}" )

# Flag used for building position independent code. (The same as that of the
# C++ compiler.)
set( CMAKE_SYCL_COMPILE_OPTIONS_PIC "${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )

# Tell CMake what compiler standards it may use with SYCL.
set( CMAKE_SYCL17_STANDARD_COMPILE_OPTION "-std=c++17" )
set( CMAKE_SYCL17_EXTENSION_COMPILE_OPTION "-std=c++17" )

# Set up the variable controlling what target(s) to compile code for.
set( CMAKE_SYCL_TARGETS "spir64-unknown-unknown-sycldevice"
   CACHE STRING "Comma separated list of SYCL targets" )
mark_as_advanced( CMAKE_SYCL_TARGETS )

# Set up C++17 by default.
set( CMAKE_SYCL_STANDARD 17 CACHE STRING "C++ standard to use with SYCL" )
set_property( CACHE CMAKE_SYCL_STANDARD PROPERTY STRINGS 17 )

# Configure variables set in this file for fast reload later on.
configure_file( ${CMAKE_CURRENT_LIST_DIR}/CMakeSYCLCompiler.cmake.in
   ${CMAKE_PLATFORM_INFO_DIR}/CMakeSYCLCompiler.cmake )
