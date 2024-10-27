/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/metal/device_wrapper.hpp"
#include "get_device.h"
#include "opaque_device.h"
#include "vecmem/utils/debug.hpp"

// System import(s).
#include <Metal/Metal.h>

// System include(s).
#include <cassert>

namespace vecmem::metal {

device_wrapper::device_wrapper()
    : m_device(nullptr),
      m_managedDevice(std::make_unique<details::opaque_device>()) {

    // Create a (default) new Metal device.
    m_managedDevice->m_device = MTLCreateSystemDefaultDevice();
    assert(m_managedDevice->m_device != nil);

    // Grab a bare pointer to the object.
    m_device = (__bridge void*)m_managedDevice->m_device;

    // Tell the user what happened.
    VECMEM_DEBUG_MSG(
        1, "Created an \"owning wrapper\" around device: %s",
        [[details::get_device(*this) name]
            cStringUsingEncoding:[NSString defaultCStringEncoding]]);
}

device_wrapper::device_wrapper(void* device)
    : m_device(device), m_managedDevice() {

    assert(m_device != nullptr);

    // Tell the user what happened.
    VECMEM_DEBUG_MSG(
        3, "Created a \"view wrapper\" around device: %s",
        [[details::get_device(*this) name]
            cStringUsingEncoding:[NSString defaultCStringEncoding]]);
}

device_wrapper::device_wrapper(const device_wrapper& parent)
    : m_device(nullptr), m_managedDevice() {

    // Check whether the parent owns its own device or not.
    if (parent.m_managedDevice) {
        // If so, make a copy of it, and own that copy in this object as well.
        // This makes sure that reference counts to the MTLDevice would be
        // managed properly.
        m_managedDevice =
            std::make_unique<details::opaque_device>(*(parent.m_managedDevice));
        m_device = (__bridge void*)m_managedDevice->m_device;
    } else {
        // If not, then let's just point at the same device that somebody else
        // owns.
        m_device = parent.m_device;
    }
}

device_wrapper::device_wrapper(device_wrapper&& parent)
    : m_device(nullptr), m_managedDevice(std::move(parent.m_managedDevice)) {

    // Set the bare pointer.
    if (m_managedDevice) {
        m_device = (__bridge void*)m_managedDevice->m_device;
    } else {
        m_device = parent.m_device;
    }
}

device_wrapper::~device_wrapper() {}

device_wrapper& device_wrapper::operator=(const device_wrapper& rhs) {

    // Avoid self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Check whether the copied object owns its own device or not.
    if (rhs.m_managedDevice) {
        // If so, make a copy of it, and own that copy in this object as well.
        m_managedDevice =
            std::make_unique<details::opaque_device>(*(rhs.m_managedDevice));
        m_device = (__bridge void*)m_managedDevice->m_device;
    } else {
        // If not, then let's just point at the same device that somebody else
        // owns.
        m_device = rhs.m_device;
    }

    // Return this object.
    return *this;
}

device_wrapper& device_wrapper::operator=(device_wrapper&& rhs) {

    // Avoid self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Move the managed device object.
    m_managedDevice = std::move(rhs.m_managedDevice);

    // Set the bare pointer.
    if (m_managedDevice) {
        m_device = (__bridge void*)m_managedDevice->m_device;
    } else {
        m_device = rhs.m_device;
    }

    // Return this object.
    return *this;
}

void* device_wrapper::device() {

    return m_device;
}

const void* device_wrapper::device() const {

    return m_device;
}

}  // namespace vecmem::metal
