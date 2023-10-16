/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/coalescing_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem {

/**
 * @brief This memory resource tries to allocate with several upstream resources
 * and returns the first succesful one.
 */
using coalescing_memory_resource =
    details::memory_resource_adaptor<details::coalescing_memory_resource>;

}  // namespace vecmem
