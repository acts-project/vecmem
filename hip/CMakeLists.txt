# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable HIP as a language.
enable_language( HIP )

# Project include(s).
include( vecmem-compiler-options-hip )

# External dependency/dependencies.
find_package( HIPToolkit REQUIRED )

# Set up the build of the VecMem HIP library.
vecmem_add_library( vecmem_hip hip SHARED
   # Memory management.
   "include/vecmem/memory/hip/direct_memory_manager.hpp"
   "src/memory/hip/direct_memory_manager.hip"
   # Utilities.
   "src/utils/get_device.hpp"
   "src/utils/get_device.cpp"
   "src/utils/hip_error_handling.hpp"
   "src/utils/hip_error_handling.cpp" )
target_link_libraries( vecmem_hip
   PUBLIC vecmem::core
   PRIVATE HIP::hiprt )
