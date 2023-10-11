/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/host_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem {

/**
 * @brief Memory resource which wraps standard library memory allocation calls.
 *
 * This is probably the simplest memory resource you can possibly write. It
 * is a terminal resource which does nothing but wrap @c std::aligned_alloc and
 * @c std::free. It is state-free (on the relevant levels of abstraction).
 */
using host_memory_resource =
    details::memory_resource_adaptor<details::host_memory_resource>;

}  // namespace vecmem
