# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable SYCL as a language.
enable_language( SYCL )

# Project include(s).
include( vecmem-compiler-options-sycl )

# Set up the build of the VecMem SYCL library.
vecmem_add_library( vecmem_sycl sycl SHARED
   # Memory management.
   "include/vecmem/memory/sycl/direct_memory_manager.hpp"
   "src/memory/sycl/direct_memory_manager.sycl"
   # Utilities.
   "include/vecmem/utils/sycl/device_selector.hpp"
   "src/utils/sycl/device_selector.sycl" )
target_link_libraries( vecmem_sycl PUBLIC vecmem::core )
