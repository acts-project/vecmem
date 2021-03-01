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
   # Memory management.
   "include/vecmem/memory/cuda/arena_memory_manager.hpp"
   "src/memory/cuda/arena_memory_manager.cpp"
   "include/vecmem/memory/cuda/direct_memory_manager.hpp"
   "src/memory/cuda/direct_memory_manager.cpp"
   # Utilities.
   "src/utils/cuda_error_handling.hpp"
   "src/utils/cuda_error_handling.cpp" )
target_link_libraries( vecmem_cuda
   PUBLIC vecmem::core
   PRIVATE CUDA::cudart )
