/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "get_device_name.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

// System include(s).
#include <sstream>

namespace vecmem::hip::details {

std::string get_device_name(int device) {

    // Get the device's properties.
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device) != hipSuccess) {
        return "Unknown";
    }

    // Construct a unique name out of those properties.
    std::ostringstream result;
    result << props.name << " [id: " << device << ", bus: " << props.pciBusID
           << ", device: " << props.pciDeviceID << "]";
    return result.str();
}

}  // namespace vecmem::hip::details
