/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/vecmem_hip_export.hpp"

// System include(s).
#include <memory>
#include <string>

namespace vecmem::hip {

/// Wrapper class for @c hipStream_t
///
/// It is necessary for passing around HIP stream objects in code that should
/// not be directly exposed to the HIP header(s).
///
class stream_wrapper {

public:
    /// Invalid/default device identifier
    static constexpr int INVALID_DEVICE = -1;

    /// Construct a new stream (for the specified device)
    VECMEM_HIP_EXPORT
    stream_wrapper(int device = INVALID_DEVICE);
    /// Wrap an existing @c hipStream_t object
    ///
    /// Without taking ownership of it!
    ///
    VECMEM_HIP_EXPORT
    stream_wrapper(void* stream);

    /// Copy constructor
    VECMEM_HIP_EXPORT
    stream_wrapper(const stream_wrapper& parent);
    /// Move constructor
    VECMEM_HIP_EXPORT
    stream_wrapper(stream_wrapper&& parent) noexcept;

    /// Destructor
    VECMEM_HIP_EXPORT
    ~stream_wrapper();

    /// Copy assignment
    VECMEM_HIP_EXPORT
    stream_wrapper& operator=(const stream_wrapper& rhs);
    /// Move assignment
    VECMEM_HIP_EXPORT
    stream_wrapper& operator=(stream_wrapper&& rhs) noexcept;

    /// Access a typeless pointer to the managed @c hipStream_t object
    VECMEM_HIP_EXPORT
    void* stream() const;

    /// Wait for all queued tasks from the stream to complete
    VECMEM_HIP_EXPORT
    void synchronize();

    /// Get the name of the device that the stream operates on
    VECMEM_HIP_EXPORT
    std::string device_name() const;

private:
    /// Structure holding the internals of the class
    struct impl;
    /// Pointer to the internal structure
    std::unique_ptr<impl> m_impl;

};  // class stream_wrapper

}  // namespace vecmem::hip
