# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable SYCL as a language.
enable_language( SYCL )

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-sycl )
include( vecmem-check-sycl-code-compiles )

# Set up the build of the VecMem SYCL library.
vecmem_add_library( vecmem_sycl sycl
   # Memory management.
   "include/vecmem/memory/sycl/details/memory_resource_base.hpp"
   "src/memory/memory_resource_base.sycl"
   "include/vecmem/memory/sycl/device_memory_resource.hpp"
   "src/memory/device_memory_resource.sycl"
   "include/vecmem/memory/sycl/host_memory_resource.hpp"
   "src/memory/host_memory_resource.sycl"
   "include/vecmem/memory/sycl/shared_memory_resource.hpp"
   "src/memory/shared_memory_resource.sycl"
   # Utilities.
   "include/vecmem/utils/sycl/copy.hpp"
   "src/utils/sycl/copy.sycl"
   "include/vecmem/utils/sycl/async_copy.hpp"
   "src/utils/sycl/async_copy.sycl"
   "include/vecmem/utils/sycl/queue_wrapper.hpp"
   "src/utils/sycl/queue_wrapper.sycl"
   "src/utils/sycl/get_queue.hpp"
   "src/utils/sycl/get_queue.sycl"
   "src/utils/sycl/opaque_queue.hpp" )
target_link_libraries( vecmem_sycl PUBLIC vecmem::core )

# Hide the library's symbols by default.
set_target_properties( vecmem_sycl PROPERTIES
   CXX_VISIBILITY_PRESET  "hidden"
   SYCL_VISIBILITY_PRESET "hidden" )
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_VISIBILITY_MS_COMPAT
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/empty.sycl"
   CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-fvisibility-ms-compat )
if( VECMEM_HAVE_SYCL_VISIBILITY_MS_COMPAT )
   target_compile_options( vecmem_sycl PRIVATE "-fvisibility-ms-compat" )
endif()

# Set up whether asynchronous synchronization errors should be fatal.
if( VECMEM_FAIL_ON_ASYNC_ERRORS )
   target_compile_definitions( vecmem_sycl PRIVATE VECMEM_FAIL_ON_ASYNC_ERRORS )
endif()

# Check if assertions work out of the box in the build.
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_ASSERT
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_test.sycl"
   CMAKE_FLAGS -DCMAKE_BUILD_TYPE:STRING=Debug )

# If not, check if it can be solved as described in:
#   https://github.com/intel/llvm/issues/3385
if( NOT VECMEM_HAVE_SYCL_ASSERT )

   # Check if the "CUDA polyfill" can make the test work.
   vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_CUDA_ASSERT_POLYFILL
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_test.sycl"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/sycl/cuda_assert_polyfill.sycl"
      CMAKE_FLAGS -DCMAKE_BUILD_TYPE:STRING=Debug )

   # If yes, we have to do something a bit elaborate. The "polyfill" code only
   # works correctly when linked into the binary that needs to use assertions.
   # Device code just does not work across binary boundaries. So we can't just
   # put the code into vecmem::sycl. Instead let's add a separate STATIC
   # library, which vecmem::sycl would depend on.
   if( VECMEM_HAVE_SYCL_CUDA_ASSERT_POLYFILL )

      # Set up the vecmem::sycl_polyfill library.
      vecmem_add_library( vecmem_sycl_polyfill sycl_polyfill TYPE STATIC
         "src/utils/sycl/cuda_assert_polyfill.sycl" )
      set_target_properties( vecmem_sycl_polyfill PROPERTIES
         POSITION_INDEPENDENT_CODE ON )
      target_link_libraries( vecmem_sycl INTERFACE vecmem::sycl_polyfill )

   else()
      # If not even this worked, then warn the user, and see what happens...
      message( WARNING "Assertions are not available for SYCL device code."
         " Debug builds will likely fail." )
   endif()
endif()

# Test which printf function(s) is/are available.
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_EXT_ONEAPI_PRINTF
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ext_oneapi_printf_test.sycl" )
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_ONEAPI_PRINTF
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/oneapi_printf_test.sycl" )

# Set up the appropriate flag with its help.
if( VECMEM_HAVE_SYCL_EXT_ONEAPI_PRINTF )
   target_compile_definitions( vecmem_sycl PUBLIC
      VECMEM_SYCL_PRINTF_FUNCTION=cl::sycl::ext::oneapi::experimental::printf )
elseif( VECMEM_HAVE_SYCL_ONEAPI_PRINTF )
   target_compile_definitions( vecmem_sycl PUBLIC
      VECMEM_SYCL_PRINTF_FUNCTION=cl::sycl::ONEAPI::experimental::printf )
else()
   message( WARNING "No valid printf function found for SYCL."
      " Enabling debug messages will likely not work in device code." )
   target_compile_definitions( vecmem_sycl PUBLIC
      VECMEM_SYCL_PRINTF_FUNCTION=printf VECMEM_MSG_ATTRIBUTES= )
endif()

# Test whether sycl::atomic_ref is available.
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_ATOMIC_REF
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/atomic_ref_test.sycl" )
if( VECMEM_HAVE_SYCL_ATOMIC_REF )
   target_compile_definitions( vecmem_sycl PUBLIC
      VECMEM_HAVE_SYCL_ATOMIC_REF )
endif()

# Check if sycl::queue::memset is available, and set a compiler option
# accordingly.
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_MEMSET
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/memset_test.sycl" )
if( VECMEM_HAVE_SYCL_MEMSET )
   target_compile_definitions( vecmem_sycl PRIVATE VECMEM_HAVE_SYCL_MEMSET )
endif()

# Check if sycl::local_accessor is available.
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_LOCAL_ACCESSOR
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/local_accessor_test.sycl" )
if( VECMEM_HAVE_SYCL_LOCAL_ACCESSOR )
   target_compile_definitions( vecmem_sycl INTERFACE
      VECMEM_HAVE_SYCL_LOCAL_ACCESSOR )
endif()

# Test the public headers of vecmem::sycl.
if( BUILD_TESTING AND VECMEM_BUILD_TESTING )
   file( GLOB_RECURSE vecmem_sycl_public_headers
      RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "include/*/*.hpp" )
   vecmem_test_public_headers( vecmem_sycl ${vecmem_sycl_public_headers} )
endif()
