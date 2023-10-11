/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/binary_page_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem {

/**
 * @brief A memory manager using power-of-two pages that can be split to
 * deal with allocation requests of various sizes.
 *
 * This is a non-terminal memory resource which relies on an upstream
 * allocator to do the actual allocation. The allocator will allocate only
 * large blocks with sizes power of two from the upstream allocator. These
 * blocks can then be split in half and allocated, split in half again. This
 * creates a binary tree of pages which can be either vacant, occupied, or
 * split.
 */
using binary_page_memory_resource =
    details::memory_resource_adaptor<details::binary_page_memory_resource>;

}  // namespace vecmem
