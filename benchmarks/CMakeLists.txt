# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )

# Set up Google Benchmark.
option(
   VECMEM_SETUP_GOOGLEBENCHMARK
   "Set up the Google Benchmark targets explicitly"
   TRUE
)

option(
   VECMEM_USE_SYSTEM_GOOGLEBENCHMARK
   "Pick up an existing installation of Google Benchmark from the build environment"
   FALSE
)

if(VECMEM_SETUP_GOOGLEBENCHMARK)
   if(VECMEM_USE_SYSTEM_GOOGLEBENCHMARK)
      find_package(benchmark REQUIRED)
   else()
      add_subdirectory(googlebenchmark)
   endif()
endif()

# Build a common, helper library.
# add_library( vecmem_benchmarking_common STATIC
#    "common/memory_resource_name_gen.hpp"
#    "common/memory_resource_name_gen.cpp"
#    "common/memory_resource_test_basic.hpp"
#    "common/memory_resource_test_basic.ipp"
#    "common/memory_resource_test_host_accessible.hpp"
#    "common/memory_resource_test_host_accessible.ipp"
#    "common/memory_resource_test_stress.hpp"
#    "common/memory_resource_test_stress.ipp" )
# target_link_libraries( vecmem_testing_common
#    PUBLIC vecmem::core GTest::gtest )

# Add the common benchmarking functionality.
# add_subdirectory(common)

# Include the library specific tests.
add_subdirectory(core)

if(VECMEM_BUILD_CUDA_LIBRARY)
   add_subdirectory(cuda)
endif()

if(VECMEM_BUILD_HIP_LIBRARY)
   add_subdirectory(hip)
endif()

if(VECMEM_BUILD_SYCL_LIBRARY)
   add_subdirectory(sycl)
endif()
