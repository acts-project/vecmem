# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CUDAToolkit requires CMake 3.17.
cmake_minimum_required( VERSION 3.17 )

# Project include(s).
include( vecmem-compiler-options-cpp )

# Set up the build of the VecMem Vitis library.
vecmem_add_library(vecmem_vitis vitis
   # Memory resources.
   "include/vecmem/memory/vitis/device_memory_resource.hpp"
   "src/memory/device_memory_resource.cpp"
   "src/memory/host_memory_resource.cpp"
   #   "include/vecmem/memory/vitis/managed_memory_resource.hpp"
   #   "src/memory/managed_memory_resource.cpp"
   # Utilities.
   "include/vecmem/utils/vitis/copy.hpp"
   "src/utils/vitis/copy.cpp"
   #   "include/vecmem/utils/vitis/async_copy.hpp"
   #   "src/utils/vitis/async_copy.cpp"
   #   "src/utils/vitis_error_handling.hpp"
   #   "src/utils/vitis_error_handling.cpp"
   #   "src/utils/vitis_wrappers.hpp"
   #   "src/utils/vitis_wrappers.cpp"
   #   "src/utils/get_device_name.hpp"
   #   "src/utils/get_device_name.cpp"
   #   "src/utils/select_device.hpp"
   #   "src/utils/select_device.cpp"
   #   "include/vecmem/utils/vitis/stream_wrapper.hpp"
   #   "src/utils/stream_wrapper.cpp"
   #   "src/utils/opaque_stream.hpp"
   #   "src/utils/opaque_stream.cpp" 
   )
target_link_libraries( vecmem_vitis
   PUBLIC vecmem::core
   )

# Hide the library's symbols by default.
set_target_properties( vecmem_vitis PROPERTIES
   CXX_VISIBILITY_PRESET  "hidden" )

# Set up whether asynchronous synchronization errors should be fatal.
if( VECMEM_FAIL_ON_ASYNC_ERRORS )
   target_compile_definitions( vecmem_vitis PRIVATE VECMEM_FAIL_ON_ASYNC_ERRORS )
endif()

# Test the public headers of vecmem::cuda.
if( BUILD_TESTING AND VECMEM_BUILD_TESTING )
   file( GLOB_RECURSE vecmem_vitis_public_headers
      RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "include/*/*.hpp" )
   vecmem_test_public_headers( vecmem_vitis ${vecmem_vitis_public_headers} )
endif()
