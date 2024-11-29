/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/metal/device_wrapper.hpp"

namespace vecmem::metal::details {

/// Helper function for getting a @c MTLDevice out of
/// @c vecmem::metal::device_wrapper
id get_device(const device_wrapper& device);

}  // namespace vecmem::metal::details
