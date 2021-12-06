/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <type_traits>

#include "vecmem/memory/details/unique_alloc_deleter_impl.hpp"

namespace vecmem::details {
template <typename T>
using unique_alloc_deleter = std::enable_if_t<
    std::is_trivially_constructible_v<std::remove_extent_t<T>> &&
        std::is_trivially_destructible_v<std::remove_extent_t<T>>,
    unique_alloc_deleter_impl>;
}
