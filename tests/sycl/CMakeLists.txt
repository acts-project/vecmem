# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Tests for the SYCL library's custom memory management.
vecmem_add_test( sycl_allocators test_sycl_allocators.cpp
   LINK_LIBRARIES vecmem::core vecmem::sycl )
