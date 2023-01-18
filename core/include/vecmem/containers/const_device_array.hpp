/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/device_array.hpp"

namespace vecmem {

/// Class mimicking a host-filled constant @c std::array in "device code"
template <typename T, std::size_t N>
using const_device_array = device_array<const T, N>;

}  // namespace vecmem
