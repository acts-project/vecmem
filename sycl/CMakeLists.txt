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
   "include/vecmem/memory/sycl/details/memory_resource_base.hpp"
   "src/memory/sycl/details/memory_resource_base.sycl"
   "include/vecmem/memory/sycl/device_memory_resource.hpp"
   "src/memory/sycl/device_memory_resource.sycl"
   "include/vecmem/memory/sycl/host_memory_resource.hpp"
   "src/memory/sycl/host_memory_resource.sycl"
   "include/vecmem/memory/sycl/shared_memory_resource.hpp"
   "src/memory/sycl/shared_memory_resource.sycl"
   # Utilities.
   "include/vecmem/utils/sycl/queue_wrapper.hpp"
   "src/utils/sycl/queue_wrapper.sycl"
   "src/utils/sycl/device_selector.hpp"
   "src/utils/sycl/device_selector.sycl"
   "src/utils/sycl/get_queue.hpp"
   "src/utils/sycl/get_queue.sycl"
   "src/utils/sycl/opaque_queue.hpp" )
target_link_libraries( vecmem_sycl PUBLIC vecmem::core )
