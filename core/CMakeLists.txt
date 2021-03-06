# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )

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
   "include/vecmem/containers/data/jagged_vector_buffer.hpp"
   "include/vecmem/containers/impl/jagged_vector_buffer.ipp"
   "include/vecmem/containers/data/jagged_vector_data.hpp"
   "include/vecmem/containers/impl/jagged_vector_data.ipp"
   "include/vecmem/containers/data/jagged_vector_view.hpp"
   "include/vecmem/containers/impl/jagged_vector_view.ipp"
   "include/vecmem/containers/data/vector_buffer.hpp"
   "include/vecmem/containers/impl/vector_buffer.ipp"
   "include/vecmem/containers/data/vector_view.hpp"
   "include/vecmem/containers/impl/vector_view.ipp"
   # Iterator types.
   "include/vecmem/containers/details/jagged_device_vector_iterator.hpp"
   "include/vecmem/containers/impl/jagged_device_vector_iterator.ipp"
   "include/vecmem/containers/details/reverse_iterator.hpp"
   "include/vecmem/containers/impl/reverse_iterator.ipp"
   # Allocator
   "include/vecmem/memory/allocator.hpp"
   "include/vecmem/memory/impl/allocator.ipp"
   "src/memory/allocator.cpp"
   "include/vecmem/memory/deallocator.hpp"
   "src/memory/deallocator.cpp"
   # Memory management.
   "include/vecmem/memory/atomic.hpp"
   "include/vecmem/memory/impl/atomic.ipp"
   "include/vecmem/memory/polymorphic_allocator.hpp"
   "include/vecmem/memory/memory_resource.hpp"
   "src/memory/host_memory_resource.cpp"
   "include/vecmem/memory/host_memory_resource.hpp"
   "src/memory/binary_page_memory_resource.cpp"
   "include/vecmem/memory/binary_page_memory_resource.hpp"
   "src/memory/contiguous_memory_resource.cpp"
   "include/vecmem/memory/contiguous_memory_resource.hpp"
   # Utilities.
   "include/vecmem/utils/copy.hpp"
   "include/vecmem/utils/impl/copy.ipp"
   "src/utils/copy.cpp"
   "include/vecmem/utils/debug.hpp"
   "include/vecmem/utils/type_traits.hpp"
   "include/vecmem/utils/types.hpp" )

# Add definitions necessary for the correct functioning of VECMEM_DEBUG_MSG.
string( LENGTH "${CMAKE_SOURCE_DIR}/" VECMEM_SOURCE_DIR_LENGTH )
target_compile_definitions( vecmem_core PUBLIC
   $<BUILD_INTERFACE:VECMEM_DEBUG_MSG_LVL=${VECMEM_DEBUG_MSG_LVL}>
   $<BUILD_INTERFACE:VECMEM_SOURCE_DIR_LENGTH=${VECMEM_SOURCE_DIR_LENGTH}> )
