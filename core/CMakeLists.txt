# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-check-sycl-code-compiles )

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
   # Memory management.
   "include/vecmem/memory/details/memory_resource_base.hpp"
   "src/memory/details/memory_resource_base.cpp"
   "include/vecmem/memory/atomic.hpp"
   "include/vecmem/memory/impl/atomic.ipp"
   "include/vecmem/memory/device_atomic_ref.hpp"
   "include/vecmem/memory/impl/device_atomic_ref.ipp"
   "include/vecmem/memory/polymorphic_allocator.hpp"
   "include/vecmem/memory/memory_resource.hpp"
   "src/memory/alignment.hpp"
   "src/memory/arena.hpp"
   "src/memory/arena.cpp"
   "src/memory/arena_memory_resource.cpp"
   "include/vecmem/memory/arena_memory_resource.hpp"
   "src/memory/identity_memory_resource.cpp"
   "include/vecmem/memory/identity_memory_resource.hpp"
   "src/memory/terminal_memory_resource.cpp"
   "include/vecmem/memory/terminal_memory_resource.hpp"
   "src/memory/host_memory_resource.cpp"
   "include/vecmem/memory/host_memory_resource.hpp"
   "src/memory/binary_page_memory_resource.cpp"
   "src/memory/binary_page_memory_resource_impl.hpp"
   "src/memory/binary_page_memory_resource_impl.cpp"
   "include/vecmem/memory/binary_page_memory_resource.hpp"
   "src/memory/contiguous_memory_resource.cpp"
   "include/vecmem/memory/contiguous_memory_resource.hpp"
   "src/memory/instrumenting_memory_resource.cpp"
   "include/vecmem/memory/instrumenting_memory_resource.hpp"
   "src/memory/choice_memory_resource.cpp"
   "include/vecmem/memory/choice_memory_resource.hpp"
   "src/memory/coalescing_memory_resource.cpp"
   "include/vecmem/memory/coalescing_memory_resource.hpp"
   "src/memory/conditional_memory_resource.cpp"
   "include/vecmem/memory/conditional_memory_resource.hpp"
   "src/memory/debug_memory_resource.cpp"
   "include/vecmem/memory/debug_memory_resource.hpp"
   "include/vecmem/memory/details/unique_alloc_deleter.hpp"
   "include/vecmem/memory/details/unique_obj_deleter.hpp"
   "include/vecmem/memory/unique_ptr.hpp"
   "include/vecmem/memory/details/is_aligned.hpp"
   "src/memory/details/is_aligned.cpp"
   # Utilities.
   "include/vecmem/utils/copy.hpp"
   "include/vecmem/utils/impl/copy.ipp"
   "src/utils/copy.cpp"
   "include/vecmem/utils/debug.hpp"
   "src/utils/memory_monitor.cpp"
   "include/vecmem/utils/memory_monitor.hpp"
   "include/vecmem/utils/type_traits.hpp"
   "include/vecmem/utils/types.hpp" )

# Hide the library's symbols by default.
set_target_properties( vecmem_core PROPERTIES
   CXX_VISIBILITY_PRESET "hidden" )

# Add definitions necessary for the correct functioning of VECMEM_DEBUG_MSG.
string( LENGTH "${CMAKE_SOURCE_DIR}/" VECMEM_SOURCE_DIR_LENGTH )
target_compile_definitions( vecmem_core PUBLIC
   VECMEM_DEBUG_MSG_LVL=${VECMEM_DEBUG_MSG_LVL}
   $<BUILD_INTERFACE:VECMEM_SOURCE_DIR_LENGTH=${VECMEM_SOURCE_DIR_LENGTH}> )

# The library headers make checks on the value of the __cplusplus macro. So we
# need to make sure that MSVC would always set that macro up when using this
# library.
target_compile_options( vecmem_core PUBLIC
   $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/Zc:__cplusplus>
   $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler /Zc:__cplusplus> )

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
   target_sources( vecmem_core
      PRIVATE "src/memory/default_resource_polyfill.cpp" )
   source_group( "src\\\\memory"
      FILES "src/memory/default_resource_polyfill.cpp" )
endif()

# Figure out how to use various SYCL features.
if( VECMEM_BUILD_SYCL_LIBRARY )

   # Test which printf function(s) is/are available.
   vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_EXT_ONEAPI_PRINTF
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ext_oneapi_printf_test.sycl" )
   vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_ONEAPI_PRINTF
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/oneapi_printf_test.sycl" )

   # Set up the appropriate flag with its help.
   if( VECMEM_HAVE_SYCL_EXT_ONEAPI_PRINTF )
      target_compile_definitions( vecmem_core PUBLIC
         VECMEM_SYCL_PRINTF_FUNCTION=cl::sycl::ext::oneapi::experimental::printf )
   elseif( VECMEM_HAVE_SYCL_ONEAPI_PRINTF )
      target_compile_definitions( vecmem_core PUBLIC
         VECMEM_SYCL_PRINTF_FUNCTION=cl::sycl::ONEAPI::experimental::printf )
   else()
      message( WARNING "No valid printf function found for SYCL."
         " Enabling debug messages will likely not work." )
      target_compile_definitions( vecmem_core PUBLIC
         VECMEM_ONEAPI_PRINTF_FUNCTION=printf )
   endif()

   # Test whether sycl::atomic_ref is available.
   vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_ATOMIC_REF
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/atomic_ref_test.sycl" )
   if( VECMEM_HAVE_SYCL_ATOMIC_REF )
      target_compile_definitions( vecmem_core PUBLIC
         VECMEM_HAVE_SYCL_ATOMIC_REF )
   endif()
endif()

# Test whether we have CLZ builtins.
check_cxx_source_compiles( "
   #include <intrin.h>
   #include <cstddef>
   int main() {
      (void)_lzcnt_u64(static_cast<std::size_t>(1UL));
      return 0;
   }
   " VECMEM_HAVE_LZCNT_U64 )
if( VECMEM_HAVE_LZCNT_U64 )
   target_compile_definitions(
      vecmem_core
      PRIVATE VECMEM_HAVE_LZCNT_U64
   )
endif()

check_cxx_source_compiles( "
   #include <cstddef>
   int main() {
      (void)__builtin_clzl(static_cast<std::size_t>(1UL));
      return 0;
   }
   " VECMEM_HAVE_BUILTIN_CLZL )
if( VECMEM_HAVE_BUILTIN_CLZL )
   target_compile_definitions(
      vecmem_core
      PRIVATE VECMEM_HAVE_BUILTIN_CLZL
   )
endif()

# Check if std::aligned_alloc is available. With MSVC (at the time of writing)
# it is not.
include( CheckCXXSymbolExists )
check_cxx_symbol_exists( "std::aligned_alloc" "cstdlib"
   VECMEM_HAVE_STD_ALIGNED_ALLOC )
if( VECMEM_HAVE_STD_ALIGNED_ALLOC )
   target_compile_definitions( vecmem_core
      PRIVATE VECMEM_HAVE_STD_ALIGNED_ALLOC )
endif()

# Test the public headers of vecmem::core.
if( BUILD_TESTING AND VECMEM_BUILD_TESTING )
   file( GLOB_RECURSE vecmem_core_public_headers
      RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "include/*/*.hpp" )
   vecmem_test_public_headers( vecmem_core ${vecmem_core_public_headers} )
endif()
