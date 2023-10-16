/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/cuda/details/host_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem::cuda {

/// Memory resource that wraps page-locked CUDA host allocation.
using host_memory_resource =
    vecmem::details::memory_resource_adaptor<details::host_memory_resource>;

}  // namespace vecmem::cuda
