# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the build of the VecMem core library.
vecmem_add_library( vecmem_core core SHARED
   # STL allocators.
   "include/vecmem/allocators/allocator.hpp"
   # STL mimicking containers.
   "include/vecmem/containers/const_device_vector.hpp"
   "include/vecmem/containers/device_vector.hpp"
   "include/vecmem/containers/host_vector.hpp"
   "include/vecmem/containers/static_vector.hpp"
   # Memory management.
   "include/vecmem/memory/memory_manager_interface.hpp"
   "include/vecmem/memory/memory_manager.hpp"
   "src/memory/memory_manager.cpp"
   "include/vecmem/memory/host_memory_manager.hpp"
   "src/memory/host_memory_manager.cpp"
   # Utilities.
   "include/vecmem/utils/types.hpp" )
