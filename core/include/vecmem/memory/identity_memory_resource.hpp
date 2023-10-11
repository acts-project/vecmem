/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/identity_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem {

/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource.
 *
 * This allocator is here to act as the unit in the monoid of memory resources.
 * It serves only a niche practical purpose.
 */
using identity_memory_resource =
    details::memory_resource_adaptor<details::identity_memory_resource>;

}  // namespace vecmem
