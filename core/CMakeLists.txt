# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the build of the VecMem core library.
vecmem_add_library( vecmem_core core SHARED
   # STL mimicking containers.
   "include/vecmem/containers/array.hpp"
   "include/vecmem/containers/impl/array.ipp"
   "include/vecmem/containers/const_device_array.hpp"
   "include/vecmem/containers/const_device_vector.hpp"
   "include/vecmem/containers/device_array.hpp"
   "include/vecmem/containers/impl/device_array.ipp"
   "include/vecmem/containers/device_vector.hpp"
   "include/vecmem/containers/impl/device_vector.ipp"
   "include/vecmem/containers/static_vector.hpp"
   "include/vecmem/containers/impl/static_vector.ipp"
   "include/vecmem/containers/jagged_device_vector.hpp"
   "include/vecmem/containers/impl/jagged_device_vector.ipp"
   "include/vecmem/containers/jagged_vector.hpp"
   "include/vecmem/containers/impl/jagged_vector.ipp"
   "include/vecmem/containers/vector.hpp"
   "include/vecmem/containers/impl/vector.ipp"
   # Data holding/transporting types.
   "include/vecmem/containers/data/jagged_vector_data.hpp"
   "include/vecmem/containers/impl/jagged_vector_data.ipp"
   "include/vecmem/containers/data/jagged_vector_view.hpp"
   "include/vecmem/containers/impl/jagged_vector_view.ipp"
   "include/vecmem/containers/data/vector_buffer.hpp"
   "include/vecmem/containers/impl/vector_buffer.ipp"
   "include/vecmem/containers/data/vector_view.hpp"
   "include/vecmem/containers/impl/vector_view.ipp"
   # Allocator
   "include/vecmem/memory/allocator.hpp"
   "include/vecmem/memory/allocator.ipp"
   "src/memory/allocator.cpp"
   "include/vecmem/memory/deallocator.hpp"
   "src/memory/deallocator.cpp"
   # Memory management.
   "include/vecmem/memory/polymorphic_allocator.hpp"
   "include/vecmem/memory/memory_resource.hpp"
   "src/memory/host_memory_resource.cpp"
   "include/vecmem/memory/host_memory_resource.hpp"
   "src/memory/binary_page_memory_resource.cpp"
   "include/vecmem/memory/binary_page_memory_resource.hpp"
   "src/memory/contiguous_memory_resource.cpp"
   "include/vecmem/memory/contiguous_memory_resource.hpp"
   # Utilities.
   "include/vecmem/containers/details/jagged_device_vector_iterator.hpp"
   "include/vecmem/containers/impl/jagged_device_vector_iterator.ipp"
   "include/vecmem/utils/reverse_iterator.hpp"
   "include/vecmem/utils/reverse_iterator.ipp"
   "include/vecmem/utils/type_traits.hpp"
   "include/vecmem/utils/types.hpp" )
