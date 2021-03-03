# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Tests for the core library's custom allocators.
vecmem_add_test( core_allocators test_core_allocators.cpp
   LINK_LIBRARIES vecmem::core )

# Tests for the core library's container types.
vecmem_add_test( core_arrays test_core_arrays.cpp
   LINK_LIBRARIES vecmem::core )
vecmem_add_test( core_containers test_core_containers.cpp
   LINK_LIBRARIES vecmem::core )

# Tests for the core library's binary page manager.
vecmem_add_test( core_binary_page test_core_binary_page.cpp
   LINK_LIBRARIES vecmem::core )

# Tests for the contiguous memory resource
vecmem_add_test( core_contiguous_memory_resource test_core_contiguous_memory_resource.cpp
   LINK_LIBRARIES vecmem::core )
