# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable SYCL as a language.
enable_language( SYCL )

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-sycl )

# Test all of the SYCL library's features.
vecmem_add_test( sycl
   "test_sycl_memory_resources.cpp"
   "test_sycl_containers.sycl" "test_sycl_jagged_containers.sycl"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../sycl/src/utils/sycl/device_selector.hpp"
   "${CMAKE_CURRENT_SOURCE_DIR}/../../sycl/src/utils/sycl/device_selector.sycl"
   LINK_LIBRARIES vecmem::core vecmem::sycl GTest::gtest_main
                  vecmem_testing_common )
