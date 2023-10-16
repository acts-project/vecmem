/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/cuda/details/device_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

/// @brief Namespace holding types that work on/with CUDA
namespace vecmem::cuda {

/// Memory resource that wraps direct allocations on a CUDA device.
using device_memory_resource =
    vecmem::details::memory_resource_adaptor<details::device_memory_resource>;

}  // namespace vecmem::cuda
