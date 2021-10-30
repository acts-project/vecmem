# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( vecmem-compiler-options-cpp )

# Set up GoogleTest.
option( VECMEM_SETUP_GOOGLETEST "Set up the GoogleTest targets explicitly"
   TRUE )
option( VECMEM_USE_SYSTEM_GOOGLETEST
   "Pick up an existing installation of GoogleTest from the build environment"
   FALSE )
if( VECMEM_SETUP_GOOGLETEST )
   if( VECMEM_USE_SYSTEM_GOOGLETEST )
      find_package( GTest REQUIRED )
   else()
      add_subdirectory( googletest )
   endif()
endif()

# Build a common, helper library.
add_library( vecmem_testing_common STATIC
   "common/memory_resource_name_gen.hpp"
   "common/memory_resource_name_gen.cpp"
   "common/memory_resource_test_basic.hpp"
   "common/memory_resource_test_basic.cpp"
   "common/memory_resource_test_host_accessible.hpp"
   "common/memory_resource_test_host_accessible.cpp"
   "common/memory_resource_test_stress.hpp"
   "common/memory_resource_test_stress.cpp" )
target_link_libraries( vecmem_testing_common
   PUBLIC vecmem::core GTest::gtest )

# Include the library specific tests.
add_subdirectory( core )
if( VECMEM_BUILD_CUDA_LIBRARY )
   add_subdirectory( cuda )
endif()
if( VECMEM_BUILD_HIP_LIBRARY )
   add_subdirectory( hip )
endif()
if( VECMEM_BUILD_SYCL_LIBRARY )
   add_subdirectory( sycl )
endif()
