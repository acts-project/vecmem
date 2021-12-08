# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Google Benchmark as part of the VecMem project" )

# Declare where to get googlebenchmark from.
FetchContent_Declare( googlebenchmark
   URL "https://github.com/google/benchmark/archive/refs/tags/v1.6.0.tar.gz"
   URL_MD5 "a7cb118b00430e22cb16774a28fce7ec" )

# Option(s) used in the build of Google Benchmark.
set( BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
   "Enable/disable testing of the benchmark library" )

# Get it into the current directory.
FetchContent_Populate( googlebenchmark )
add_subdirectory( "${googlebenchmark_SOURCE_DIR}"
   "${googlebenchmark_BINARY_DIR}" EXCLUDE_FROM_ALL )
