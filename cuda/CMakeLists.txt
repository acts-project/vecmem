# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# Set up the build of the VecMem CUDA library.
vecmem_add_library( vecmem_cuda cuda SHARED
   # Memory management.
   "include/vecmem/memory/cuda/arena_memory_manager.hpp"
   "src/memory/cuda/arena_memory_manager.cu"
   "include/vecmem/memory/cuda/direct_memory_manager.hpp"
   "src/memory/cuda/direct_memory_manager.cu"
   # Utilities.
   "include/vecmem/utils/cuda_error_handling.hpp"
   "src/utils/cuda_error_handling.cu" )
target_link_libraries( vecmem_cuda PUBLIC vecmem::core )
