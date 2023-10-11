/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/debug_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem {

/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource, but alerts the user of potential problems.
 *
 * For example, this memory resource can be used to catch overlapping
 * allocations, double frees, invalid frees, and other memory integrity issues.
 */
using debug_memory_resource =
    details::memory_resource_adaptor<details::debug_memory_resource>;

}  // namespace vecmem
