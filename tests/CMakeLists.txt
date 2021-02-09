# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the library specific tests.
add_subdirectory( core )
if( VECMEM_BUILD_CUDA_LIBRARY )
   add_subdirectory( cuda )
endif()
if( VECMEM_BUILD_SYCL_LIBRARY )
   add_subdirectory( sycl )
endif()
