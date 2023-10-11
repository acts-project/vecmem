/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/contiguous_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem {

/**
 * @brief Downstream allocator that ensures that allocations are contiguous.
 *
 * When programming for co-processors, it is often desriable to keep
 * allocations contiguous. This downstream allocator fills that need. When
 * configured with an upstream memory resource, it will start out by
 * allocating a single, large, chunk of memory from the upstream. Then, it
 * will hand out pointers along that memory in a contiguous fashion. This
 * allocator guarantees that each consecutive allocation will start right at
 * the end of the previous.
 *
 * @note The allocation size on the upstream allocator is also the maximum
 * amount of memory that can be allocated from the contiguous memory
 * resource.
 */
using contiguous_memory_resource =
    details::memory_resource_adaptor<details::contiguous_memory_resource>;

}  // namespace vecmem
