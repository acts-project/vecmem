# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the build of the VecMem core library.
vecmem_add_library( vecmem_core core SHARED
   # STL mimicking containers.
   "include/vecmem/containers/array.hpp"
   "include/vecmem/containers/array.ipp"
   "include/vecmem/containers/const_device_array.hpp"
   "include/vecmem/containers/const_device_vector.hpp"
   "include/vecmem/containers/device_array.hpp"
   "include/vecmem/containers/device_array.ipp"
   "include/vecmem/containers/device_vector.hpp"
   "include/vecmem/containers/device_vector.ipp"
   "include/vecmem/containers/static_vector.hpp"
   "include/vecmem/containers/vector.hpp"
   "include/vecmem/containers/vector.ipp"
   # Implementation details for the containers.
   "include/vecmem/containers/details/vector_data.hpp"
   "include/vecmem/containers/details/vector_data.ipp"
   # Memory management.
   "include/vecmem/memory/resources/memory_resource.hpp"
   "src/memory/host_memory_resource.cpp"
   "include/vecmem/memory/host_memory_resource.hpp"
   "src/memory/binary_page_memory_resource.cpp"
   "include/vecmem/memory/binary_page_memory_resource.hpp"
   "src/memory/contiguous_memory_resource.cpp"
   "include/vecmem/memory/contiguous_memory_resource.hpp"
   # Utilities.
   "include/vecmem/utils/deleter.hpp"
   "src/utils/deleter.cpp"
   "include/vecmem/utils/reverse_iterator.hpp"
   "include/vecmem/utils/reverse_iterator.ipp"
   "include/vecmem/utils/types.hpp" )
