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
include( vecmem-check-sycl-code-compiles )

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

# Check if assertions work out of the box in the build.
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_ASSERT
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_test.sycl" )

# If not, check if it can be solved as described in:
#   https://github.com/intel/llvm/issues/3385
if( NOT VECMEM_HAVE_SYCL_ASSERT )

   # Check if the "CUDA polyfill" can make the test work.
   vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_CUDA_ASSERT_POLYFILL
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_test.sycl"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/sycl/cuda_assert_polyfill.sycl" )

   # If yes, we have to do something a bit elaborate. The "polyfill" code only
   # works correctly when linked into the binary that needs to use assertions.
   # Device code just does not work across binary boundaries. So we can't just
   # put the code into vecmem::sycl. Instead let's add a separate STATIC
   # library, which vecmem::sycl would depend on.
   if( VECMEM_HAVE_SYCL_CUDA_ASSERT_POLYFILL )

      # Set up the vecmem::sycl_polyfill library.
      vecmem_add_library( vecmem_sycl_polyfill sycl_polyfill TYPE STATIC
         "src/utils/sycl/cuda_assert_polyfill.sycl" )
      target_link_libraries( vecmem_sycl INTERFACE vecmem::sycl_polyfill )

   else()
      # If not even this worked, then warn the user, and see what happens...
      message( WARNING "Assertions are not available for SYCL device code."
         " Debug builds will likely fail." )
   endif()
endif()

# Check if sycl::queue::memset is available, and set a compiler option
# accordingly.
vecmem_check_sycl_code_compiles( VECMEM_HAVE_SYCL_MEMSET
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/memset_test.sycl" )
if( VECMEM_HAVE_SYCL_MEMSET )
   target_compile_definitions( vecmem_sycl PRIVATE VECMEM_HAVE_SYCL_MEMSET )
endif()

# Check if a definitely error-free source file can be compiled successfully with
# -Werror, and only add it to the debug builds if it can be.
vecmem_check_sycl_code_compiles( VECMEM_SYCL_WERROR_USABLE
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/warning_test.sycl"
   COMPILE_DEFINITIONS "-Werror" )
if( VECMEM_SYCL_WERROR_USABLE )
   vecmem_add_flag( CMAKE_SYCL_FLAGS_DEBUG "-Werror" )
endif()
