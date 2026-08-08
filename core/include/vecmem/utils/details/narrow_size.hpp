/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace vecmem::details {

/// Narrow a host-side element count to a device-side size type.
///
/// The device-side containers deliberately use a 32-bit @c size_type rather
/// than @c std::size_t. Resizable vectors and buffers update their size with
/// atomic operations on the device, and CUDA, HIP and SYCL only guarantee
/// atomic support for 32-bit integer types. See acts-project/vecmem#96.
///
/// Sizes therefore have to be narrowed when they cross from the host into a
/// device-side object. That narrowing is intentional, but it should be
/// explicit at every call site rather than implicit, so that it is visible in
/// review and so that the build stays free of @c -Wconversion and
/// @c -Wnarrowing diagnostics. A container with more elements than the
/// device-side size type can represent is not a supported configuration, and
/// is caught by an assertion in debug builds.
///
/// Only ever called from host code: sizes are computed on the host and
/// baked into the view/buffer before it is handed to a device. Keeping it
/// host-only avoids pulling @c std::numeric_limits, which is not marked as
/// device-callable, into device compilation.
///
/// @tparam SIZE_TYPE  The device-side size type to narrow to
/// @tparam INPUT_TYPE The host-side size type to narrow from
/// @param  size       The host-side size
/// @return The same value, represented in the device-side size type
///
template <typename SIZE_TYPE, typename INPUT_TYPE>
VECMEM_HOST constexpr SIZE_TYPE narrow_size(INPUT_TYPE size) {

    static_assert(
        std::is_integral_v<SIZE_TYPE> && std::is_unsigned_v<SIZE_TYPE>,
        "The device-side size type must be an unsigned integer type");
    static_assert(
        std::is_integral_v<INPUT_TYPE> && std::is_unsigned_v<INPUT_TYPE>,
        "Only unsigned sizes can be narrowed with this helper");

    assert(static_cast<std::size_t>(size) <=
           static_cast<std::size_t>(std::numeric_limits<SIZE_TYPE>::max()));
    return static_cast<SIZE_TYPE>(size);
}

}  // namespace vecmem::details
