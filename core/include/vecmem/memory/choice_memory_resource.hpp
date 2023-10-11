/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/choice_memory_resource.hpp"
#include "vecmem/memory/details/memory_resource_adaptor.hpp"

namespace vecmem {

/**
 * @brief This memory resource conditionally allocates memory. It is
 * constructed with a function that determines which upstream resource to use.
 *
 * This resource can be used to construct complex conditional allocation
 * schemes.
 */
using choice_memory_resource =
    details::memory_resource_adaptor<details::choice_memory_resource>;

}  // namespace vecmem
