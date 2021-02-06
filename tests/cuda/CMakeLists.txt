# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Enable CUDA as a language.
enable_language( CUDA )

# Tests for the CUDA library's custom memory management.
vecmem_add_test( cuda_allocators test_cuda_allocators.cpp
   LINK_LIBRARIES vecmem::core vecmem::cuda )
