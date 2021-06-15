/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

/*
 * The purpose of this file is to provide uniform access (on a source-code
 * level) to the memory_resource type from the standard library. These are
 * either in the std::pmr namespace or in the std::experimental::pmr namespace
 * depending on the GCC version used, so we try to unify them by aliassing
 * depending on the compiler feature flags.
 */
#if __has_include(<memory_resource>)
#include <memory_resource>

namespace vecmem {
using memory_resource = std::pmr::memory_resource;
}
#elif __has_include(<experimental/memory_resource>)
#include <experimental/memory_resource>

namespace vecmem {
using memory_resource = std::experimental::pmr::memory_resource;
}
#else
#error "vecmem requires C++17 LFTS V1 (P0220R1) component memory_resource!"
#endif
