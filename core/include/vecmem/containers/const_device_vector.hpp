/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/device_vector.hpp"

namespace vecmem {

/// Class mimicking a constant @c std::vector in "device code"
template <typename T>
using const_device_vector = device_vector<const T>;

}  // namespace vecmem
