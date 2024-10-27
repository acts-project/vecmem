/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System import(s).
#include <Metal/Metal.h>

namespace vecmem::metal::details {

/// Helper struct for managing device objects in memory
struct opaque_device {
    /// The device object
    id<MTLDevice> m_device;
};

}  // namespace vecmem::metal::details
