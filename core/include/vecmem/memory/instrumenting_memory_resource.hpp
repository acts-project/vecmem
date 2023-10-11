/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/instrumenting_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem {

/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource while recording useful statistics and information
 * about these events.
 *
 * This allocator is here to allow us to debug, to profile, to test, but also
 * to instrument user code.
 */
using instrumenting_memory_resource =
    details::memory_resource_adaptor<details::instrumenting_memory_resource>;

}  // namespace vecmem
