/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>

namespace vecmem {
namespace details {

/// Hand-written implementation of a memmove function
///
/// It is not super efficient, but:
///  - It work in device code;
///  - For the small arrays that it would be used for, should be good enough.
///
/// @param dest Pointer to the destination memory
/// @param src Pointer to the source memory
/// @param bytes Number of bytes to move
///
VECMEM_HOST_AND_DEVICE
inline void memmove(void* dest, const void* src, std::size_t bytes);

}  // namespace details
}  // namespace vecmem

// Include the implementation.
#include "vecmem/utils/impl/memmove.ipp"
