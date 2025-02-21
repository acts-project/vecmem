# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# OBJCXX support requires CMake 3.16.
cmake_minimum_required( VERSION 3.16 )

# Enable the OBJCXX language.
enable_language( OBJCXX )

# Find the necessary macOS framework(s).
find_library( VECMEM_METAL_FRAMEWORK "Metal" REQUIRED )

# Test all of the Metal library's features.
vecmem_add_test( metal
   "test_metal_main.mm"
   "test_metal_memory_resources.cpp"
   LINK_LIBRARIES ${VECMEM_METAL_FRAMEWORK} GTest::gtest_main
                  vecmem::core vecmem::metal vecmem_testing_common )
