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
#include "vecmem/memory/sycl/details/host_memory_resource.hpp"

namespace vecmem::sycl {

/// Host memory resource, connected to a specific SYCL device
using host_memory_resource =
    vecmem::details::memory_resource_adaptor<details::host_memory_resource>;

}  // namespace vecmem::sycl
