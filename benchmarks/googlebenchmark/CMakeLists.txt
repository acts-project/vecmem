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

set(BENCHMARK_ENABLE_GTEST_TESTS Off)
set(BENCHMARK_ENABLE_TESTING Off)

# Get it into the current directory.
FetchContent_Populate( googlebenchmark )
add_subdirectory( "${googlebenchmark_SOURCE_DIR}" "${googlebenchmark_BINARY_DIR}"
   EXCLUDE_FROM_ALL )

# Set up aliases for the GTest targets with the same name that they have
# when we find GTest pre-installed.
# add_library( benchmark::benchmark ALIAS benchmark_benchmark )
# add_library( benchmark::benchmark_main ALIAS benchmark_benchmark_main )
