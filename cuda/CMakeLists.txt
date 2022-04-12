# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CUDAToolkit requires CMake 3.17.
cmake_minimum_required( VERSION 3.17 )

# Project include(s).
include( vecmem-compiler-options-cpp )

# External dependency/dependencies.
find_package( CUDAToolkit REQUIRED )

# Set up the build of the VecMem CUDA library.
vecmem_add_library( vecmem_cuda cuda
   # Memory resources.
   "include/vecmem/memory/cuda/device_memory_resource.hpp"
   "src/memory/cuda/device_memory_resource.cpp"
   "include/vecmem/memory/cuda/host_memory_resource.hpp"
   "src/memory/cuda/host_memory_resource.cpp"
   "include/vecmem/memory/cuda/managed_memory_resource.hpp"
   "src/memory/cuda/managed_memory_resource.cpp"
   # Utilities.
   "include/vecmem/utils/cuda/copy.hpp"
   "src/utils/cuda/copy.cpp"
   "include/vecmem/utils/cuda/async_copy.hpp"
   "src/utils/cuda/async_copy.cpp"
   "src/utils/cuda_error_handling.hpp"
   "src/utils/cuda_error_handling.cpp"
   "src/utils/cuda_wrappers.hpp"
   "src/utils/cuda_wrappers.cpp"
   "src/utils/get_device_name.hpp"
   "src/utils/get_device_name.cpp"
   "src/utils/select_device.hpp"
   "src/utils/select_device.cpp"
   "include/vecmem/utils/cuda/stream_wrapper.hpp"
   "src/utils/stream_wrapper.cpp"
   "src/utils/opaque_stream.hpp"
   "src/utils/opaque_stream.cpp" )
target_link_libraries( vecmem_cuda
   PUBLIC vecmem::core
   PRIVATE CUDA::cudart )

# Hide the library's symbols by default.
set_target_properties( vecmem_cuda PROPERTIES
   CXX_VISIBILITY_PRESET  "hidden"
   CUDA_VISIBILITY_PRESET "hidden" )
