# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Require CMake 3.13 for target_link_options(...).
cmake_minimum_required( VERSION 3.13 )

# Set up the build of the VecMem SYCL library.
vecmem_add_library( vecmem_sycl sycl SHARED
   # Memory management.
   "include/vecmem/memory/sycl/direct_memory_manager.hpp"
   "src/memory/sycl/direct_memory_manager.cpp"
   # Utilities.
   "include/vecmem/utils/sycl/device_selector.hpp"
   "src/utils/sycl/device_selector.cpp" )
target_link_libraries( vecmem_sycl PUBLIC vecmem::core )
target_compile_options( vecmem_sycl PUBLIC "-fsycl" )
target_link_options( vecmem_sycl PUBLIC "-fsycl" )
