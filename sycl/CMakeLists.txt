# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable SYCL as a language.
enable_language( SYCL )

# Project include(s).
include( vecmem-compiler-options-cpp )
include( vecmem-compiler-options-sycl )

# Set up the build of the VecMem SYCL library.
vecmem_add_library( vecmem_sycl sycl
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
   "include/vecmem/utils/sycl/copy.hpp"
   "src/utils/sycl/copy.sycl"
   "include/vecmem/utils/sycl/queue_wrapper.hpp"
   "src/utils/sycl/queue_wrapper.sycl"
   "src/utils/sycl/device_selector.hpp"
   "src/utils/sycl/device_selector.sycl"
   "src/utils/sycl/get_queue.hpp"
   "src/utils/sycl/get_queue.sycl"
   "src/utils/sycl/opaque_queue.hpp" )
target_link_libraries( vecmem_sycl PUBLIC vecmem::core )

# Hide the library's symbols by default.
set_target_properties( vecmem_sycl PROPERTIES
   CXX_VISIBILITY_PRESET  "hidden"
   SYCL_VISIBILITY_PRESET "hidden" )

# Helper function for checking if some SYCL files can be built into an
# executable.
function( check_sycl_code_compiles _variable )
   if( DEFINED ${_variable} )
      return()
   endif()
   if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
      message( STATUS "Performing Test ${_variable}" )
   else()
      message( CHECK_START "Performing Test ${_variable}" )
   endif()
   try_compile( ${_variable}
      "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${_variable}"
      SOURCES ${ARGN} )
   if( ${_variable} )
      if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
         message( STATUS "Performing Test ${_variable} - Success" )
      else()
         message( CHECK_PASS "Success" )
      endif()
   else()
      if( ${CMAKE_VERSION} VERSION_LESS 3.17 )
         message( STATUS "Performing Test ${_variable} - Failed" )
      else()
         message( CHECK_FAIL "Failed" )
      endif()
   endif()
endfunction( check_sycl_code_compiles )

# Check if assertions work out of the box in the build.
check_sycl_code_compiles( VECMEM_SYCL_HAVE_ASSERT
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_test.sycl" )

# If not, check if it can be solved as described in:
#   https://github.com/intel/llvm/issues/3385
if( NOT VECMEM_SYCL_HAVE_ASSERT )

   # Check if the "CUDA polyfill" can make the test work.
   check_sycl_code_compiles( VECMEM_SYCL_HAVE_ASSERT_POLYFILL
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_test.sycl"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/sycl/cuda_assert_polyfill.sycl" )

   # If yes, we have to do something a bit elaborate. The "polyfill" code only
   # works correctly when linked into the binary that needs to use assertions.
   # Device code just does not work across binary boundaries. So we can't just
   # put the code into vecmem::sycl. Instead let's add a separate STATIC
   # library, which vecmem::sycl would depend on.
   if( VECMEM_SYCL_HAVE_ASSERT_POLYFILL )

      # Set up the vecmem::sycl_polyfill library.
      vecmem_add_library( vecmem_sycl_polyfill sycl_polyfill TYPE STATIC
         "src/utils/sycl/cuda_assert_polyfill.sycl" )
      target_link_libraries( vecmem_sycl INTERFACE vecmem::sycl_polyfill )

   else()
      # If not even this worked, then warn the user, and see what happens...
      message( WARNING "Assertions are not available for SYCL device code."
         "The build will likely fail." )
   endif()
endif()
