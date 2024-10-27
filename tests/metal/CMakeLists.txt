# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Test all of the Metal library's features.
vecmem_add_test( metal
   "test_metal_memory_resources.cpp"
   LINK_LIBRARIES GTest::gtest_main vecmem::core vecmem::metal
                  vecmem_testing_common )
