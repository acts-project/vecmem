# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the project.
cmake_minimum_required( VERSION 3.10 )
project( vecmem VERSION 0.4.0 LANGUAGES CXX )

# Standard CMake include(s).
include( GNUInstallDirs )

# Explicitly set the output directory for the binaries. Such that if this
# project is included by another project, the main project's configuration would
# win out.
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}" CACHE PATH
   "Directory for the built binaries" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
   "Directory for the built libraries" )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
   "Directory for the built static libraries" )

# Include the VecMem CMake code.
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
include( vecmem-functions )
include( vecmem-options )

# Add the library sub-directories.
add_subdirectory( core )
if( VECMEM_BUILD_CUDA_LIBRARY )
   add_subdirectory( cuda )
endif()
if( VECMEM_BUILD_HIP_LIBRARY )
   add_subdirectory( hip )
endif()
if( VECMEM_BUILD_SYCL_LIBRARY )
   add_subdirectory( sycl )
endif()

# Set up the test(s).
option( VECMEM_BUILD_TESTING "Build the unit tests of VecMem" ON )
include( CTest )
if( BUILD_TESTING AND VECMEM_BUILD_TESTING )
  add_subdirectory( tests )
endif()

# Set up the packaging of the project.
include( vecmem-packaging )
