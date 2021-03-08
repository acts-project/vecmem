# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# Project include(s).
include( vecmem-compiler-options-cuda )

# External dependency/dependencies.
find_package( CUDAToolkit REQUIRED )

# Set up the build of the VecMem CUDA library.
vecmem_add_library( vecmem_cuda cuda SHARED
   # Memory resources.
   "include/vecmem/memory/cuda/device_memory_resource.hpp"
   "src/memory/cuda/device_memory_resource.cpp"
   "include/vecmem/memory/cuda/host_memory_resource.hpp"
   "src/memory/cuda/host_memory_resource.cpp"
   "include/vecmem/memory/cuda/managed_memory_resource.hpp"
   "src/memory/cuda/managed_memory_resource.cpp"
   # Utilities.
   "include/vecmem/utils/cuda/copy_to_device.hpp"
   "include/vecmem/utils/cuda/copy_to_device.ipp"
   "src/utils/cuda/copy_to_device.cpp"
   "include/vecmem/utils/cuda/copy_to_host.hpp"
   "include/vecmem/utils/cuda/copy_to_host.ipp"
   "src/utils/cuda/copy_to_host.cpp"
   "src/utils/cuda_error_handling.hpp"
   "src/utils/cuda_error_handling.cpp"
   "src/utils/cuda_wrappers.hpp"
   "src/utils/cuda_wrappers.cpp"
   "src/utils/select_device.hpp"
   "src/utils/select_device.cpp" )
target_link_libraries( vecmem_cuda
   PUBLIC vecmem::core
   PRIVATE CUDA::cudart )
