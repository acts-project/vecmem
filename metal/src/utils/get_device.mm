/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "get_device.h"

// System import(s).
#import <Metal/Metal.h>

// System include(s).
#include <cassert>

namespace vecmem::metal::details {

id get_device(const device_wrapper& device) {

    assert(device.device() != nullptr);
    return (__bridge id<MTLDevice>)device.device();
}

}  // namespace vecmem::metal::details
