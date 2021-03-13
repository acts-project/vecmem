# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Build GoogleTest.
include( vecmem-googletest )

# Build a common, helper library.
add_library( vecmem_testing_common STATIC
   "common/memory_resource_name_gen.hpp" "common/memory_resource_name_gen.cpp" )
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
