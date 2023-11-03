# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
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
   "test_sycl_edm_copy.cpp"
   LINK_LIBRARIES vecmem::core vecmem::sycl GTest::gtest_main
                  vecmem_testing_common )
