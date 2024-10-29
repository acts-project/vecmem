/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/vecmem_metal_export.hpp"

// System include(s).
#include <memory>

namespace vecmem::metal {

// Forward declaration(s).
namespace details {
class opaque_device;
}

/// Wrapper class for @c MTLDevice
///
/// It is necessary for passing around Metal device objects in
/// non-Objective-C(++) code.
///
class device_wrapper {

public:
    /// Construct a default device
    VECMEM_METAL_EXPORT
    device_wrapper();
    /// Wrap an existing @c MTLDevice object
    ///
    /// Without taking ownership of it!
    ///
    VECMEM_METAL_EXPORT
    device_wrapper(void* device);

    /// Copy constructor
    VECMEM_METAL_EXPORT
    device_wrapper(const device_wrapper& parent);
    /// Move constructor
    VECMEM_METAL_EXPORT
    device_wrapper(device_wrapper&& parent);

    /// Destructor
    VECMEM_METAL_EXPORT
    ~device_wrapper();

    /// Copy assignment
    VECMEM_METAL_EXPORT
    device_wrapper& operator=(const device_wrapper& rhs);
    /// Move assignment
    VECMEM_METAL_EXPORT
    device_wrapper& operator=(device_wrapper&& rhs);

    /// Access a typeless pointer to the managed @c MTLDevice object
    VECMEM_METAL_EXPORT
    void* device();
    /// Access a typeless pointer to the managed @c MTLDevice object
    VECMEM_METAL_EXPORT
    const void* device() const;

private:
    /// Bare pointer to the wrapped @c MTLDevice object
    void* m_device;

    /// Smart pointer to the managed @c MTLDevice object
    std::unique_ptr<details::opaque_device> m_managedDevice;

};  // class device_wrapper

}  // namespace vecmem::metal
