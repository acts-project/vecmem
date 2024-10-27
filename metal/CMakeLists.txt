# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# OBJCXX support requires CMake 3.16.
cmake_minimum_required( VERSION 3.16 )

# Enable the OBJCXX language.
enable_language( OBJCXX )

# Set up the build of the VecMem Metal library.
vecmem_add_library( vecmem_metal metal
   # Memory management.
   "include/vecmem/memory/metal/shared_memory_resource.hpp"
   "src/memory/shared_memory_resource.mm"
   # Utilities.
   "include/vecmem/utils/metal/device_wrapper.hpp"
   "src/utils/device_wrapper.mm"
   "src/utils/opaque_device.hpp"
   "src/utils/get_device.h"
   "src/utils/get_device.mm" )
target_link_libraries( vecmem_metal PUBLIC vecmem::core )

# Hide the library's symbols by default.
set_target_properties( vecmem_metal PROPERTIES
   OBJCXX_VISIBILITY_PRESET "hidden" )

# Link against the necessary macOS framework(s).
find_library( VECMEM_FOUNDATION_FRAMEWORK "Foundation" REQUIRED )
find_library( VECMEM_METAL_FRAMEWORK "Metal" REQUIRED )
target_link_libraries( vecmem_metal
   PRIVATE ${VECMEM_FOUNDATION_FRAMEWORK} ${VECMEM_METAL_FRAMEWORK} )
