# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable HIP as a language.
enable_language( HIP )

# Project include(s).
include(
   "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/vecmem-compiler-options-hip.cmake" )

# Set up the build of the VecMem HIP library.
vecmem_add_library( vecmem_hip hip SHARED
   # Memory management.
   "include/vecmem/memory/hip/direct_memory_manager.hpp"
   "src/memory/hip/direct_memory_manager.hip"
   # Utilities.
   "include/vecmem/utils/hip_error_handling.hpp"
   "src/utils/hip_error_handling.hip" )
target_link_libraries( vecmem_hip PUBLIC vecmem::core )
