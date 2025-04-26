/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/hip/stream_wrapper.hpp"
#include "vecmem/vecmem_hip_export.hpp"

namespace vecmem::hip {

/// Specialisation of @c vecmem::copy for HIP
///
/// This specialisation of @c vecmem::copy, unlike @c vecmem::hip::copy,
/// performs all of its operations asynchronously. Using the HIP stream
/// that is given to its constructor.
///
/// It is up to the user to ensure that copy operations are performed in the
/// right order, and they would finish before an operation that needs them
/// is executed.
///
class async_copy : public vecmem::copy {

public:
    /// Constructor with the stream to operate on
    VECMEM_HIP_EXPORT
    async_copy(const stream_wrapper& stream);
    /// Destructor
    VECMEM_HIP_EXPORT
    ~async_copy() noexcept;

protected:
    /// Perform an asynchronous memory copy using HIP
    VECMEM_HIP_EXPORT
    void do_copy(std::size_t size, const void* from, void* to,
                 type::copy_type cptype) const final;
    /// Fill a memory area using HIP asynchronously
    VECMEM_HIP_EXPORT
    void do_memset(std::size_t size, void* ptr, int value) const final;
    /// Create an event for synchronization
    VECMEM_HIP_EXPORT
    event_type create_event() const final;

private:
    /// The stream that the copies are performed on
    stream_wrapper m_stream;

};  // class async_copy

}  // namespace vecmem::hip
