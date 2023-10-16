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
#include "vecmem/memory/details/terminal_memory_resource.hpp"

namespace vecmem {

/**
 * @brief This memory resource does nothing, but it does nothing for a purpose.
 *
 * This allocator has little practical use, but can be useful for defining some
 * conditional allocation schemes.
 *
 * Reimplementation of @c std::pmr::null_memory_resource but can accept another
 * memory resource in its constructor.
 */
using terminal_memory_resource =
    details::memory_resource_adaptor<details::terminal_memory_resource>;

}  // namespace vecmem
