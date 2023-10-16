/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_adaptor.hpp"
#include "vecmem/memory/sycl/details/shared_memory_resource.hpp"

namespace vecmem::sycl {

/// Memory resource shared between the host and a specific SYCL device
using shared_memory_resource =
    vecmem::details::memory_resource_adaptor<details::shared_memory_resource>;

}  // namespace vecmem::sycl
