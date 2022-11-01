# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable SYCL as a language.
enable_language( SYCL )

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-sycl )

# Activate the -Werror flag for debug builds, based on what was found in the
# vecmem::sycl library's setup. (Here we assume that vecmem::sycl's build was
# configured before we would get here, so the variable would already be in the
# cache.)
if( VECMEM_SYCL_WERROR_USABLE )
   vecmem_add_flag( CMAKE_SYCL_FLAGS_DEBUG "-Werror" )
endif()

# Test all of the SYCL library's features.
vecmem_add_test( sycl
   "test_sycl_memory_resources.cpp"
   "test_sycl_containers.sycl" "test_sycl_jagged_containers.sycl"
   LINK_LIBRARIES vecmem::core vecmem::sycl GTest::gtest_main
                  vecmem_testing_common )
