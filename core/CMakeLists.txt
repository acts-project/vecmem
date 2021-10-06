# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )

# Set up the build of the VecMem core library.
vecmem_add_library( vecmem_core core
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
   "src/memory/arena_memory_resource.cpp"
   "include/vecmem/memory/arena_memory_resource.hpp"
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

# Hide the library's symbols by default.
set_target_properties( vecmem_core PROPERTIES
   CXX_VISIBILITY_PRESET "hidden" )

# Add definitions necessary for the correct functioning of VECMEM_DEBUG_MSG.
string( LENGTH "${CMAKE_SOURCE_DIR}/" VECMEM_SOURCE_DIR_LENGTH )
target_compile_definitions( vecmem_core PUBLIC
   $<BUILD_INTERFACE:VECMEM_DEBUG_MSG_LVL=${VECMEM_DEBUG_MSG_LVL}>
   $<BUILD_INTERFACE:VECMEM_SOURCE_DIR_LENGTH=${VECMEM_SOURCE_DIR_LENGTH}> )

# Figure out where to get <memory_resource> from.
include( CheckCXXSourceCompiles )
check_cxx_source_compiles( "
   #include <memory_resource>
   int main() {
       std::pmr::memory_resource* mr = nullptr;
       std::pmr::polymorphic_allocator<int> pa(mr);
       return 0;
   }
   " VECMEM_HAVE_PMR_MEMORY_RESOURCE )
   set(pmr_include "memory_resource")
   set(pmr_ns "std::pmr")
if( VECMEM_HAVE_PMR_MEMORY_RESOURCE )
   message( STATUS "Using memory resource types from the std::pmr namespace" )
   target_compile_definitions( vecmem_core PUBLIC
      VECMEM_HAVE_PMR_MEMORY_RESOURCE )
else()
   check_cxx_source_compiles( "
      #include <experimental/memory_resource>
      int main() {
          std::experimental::pmr::memory_resource* mr = nullptr;
          std::experimental::pmr::polymorphic_allocator<int> pa(mr);
          return 0;
      }
      " VECMEM_HAVE_EXPERIMENTAL_PMR_MEMORY_RESOURCE )
   if( VECMEM_HAVE_EXPERIMENTAL_PMR_MEMORY_RESOURCE )
      message( STATUS "Using memory resource types from the "
         "std::experimental::pmr namespace" )
      target_compile_definitions( vecmem_core PUBLIC
         VECMEM_HAVE_EXPERIMENTAL_PMR_MEMORY_RESOURCE )

      set(pmr_include "experimental/memory_resource")
      set(pmr_ns "std::experimental::pmr")
   else()
      message( SEND_ERROR
         "C++17 LFTS V1 (P0220R1) component memory_resource not found!" )
   endif()
endif()

check_cxx_source_compiles( "
   #include <${pmr_include}>
   int main() {
         ${pmr_ns}::set_default_resource(nullptr);
         (void)${pmr_ns}::get_default_resource();
         return 0;
   }
  " VECMEM_HAVE_DEFAULT_RESOURCE)

if(NOT VECMEM_HAVE_DEFAULT_RESOURCE)
   target_sources(vecmem_core PRIVATE src/memory/default_resource_polyfill.cpp)
endif()
