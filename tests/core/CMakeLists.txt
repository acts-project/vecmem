# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Test all of the core library's features.
vecmem_add_test( core
   "test_core_allocators.cpp" "test_core_arrays.cpp"
   "test_core_containers.cpp" "test_core_contiguous_memory_resource.cpp"
   "test_core_device_containers.cpp"
   "test_core_allocator.cpp"
   LINK_LIBRARIES vecmem::core GTest::gtest_main )
