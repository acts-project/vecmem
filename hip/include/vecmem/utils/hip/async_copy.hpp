/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/hip/stream_wrapper.hpp"
#include "vecmem/vecmem_hip_export.hpp"

// System include(s).
#include <memory>

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
/// Note that the class is not thread-safe. It is up to the user to ensure that
/// multiple CPU threads don't use it at the same time.
///
class async_copy : public vecmem::copy {

public:
    /// Constructor with the HIP stream to operate on
    ///
    /// @param stream The HIP stream to perform the copies in
    ///
    VECMEM_HIP_EXPORT
    async_copy(const stream_wrapper& stream);
    /// Constructor with a stream and flags to create HIP events with
    ///
    /// For details on the flags, see the documentation
    /// of @c hipEventCreateWithFlags(...) on:
    /// https://rocm.docs.amd.com/projects/HIP/en/develop/doxygen/html/group___event.html
    ///
    /// @param stream The HIP stream to perform the copies in
    /// @param event_flags Flag(s) to create internal HIP events with
    ///
    VECMEM_HIP_EXPORT
    async_copy(const stream_wrapper& stream, unsigned int event_flags);
    /// Move constructor
    VECMEM_HIP_EXPORT
    async_copy(async_copy&&) noexcept;
    /// Destructor
    VECMEM_HIP_EXPORT
    ~async_copy() noexcept;

    /// Move assignment operator
    VECMEM_HIP_EXPORT
    async_copy& operator=(async_copy&&) noexcept;

private:
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

    /// Internal data type for the class
    struct impl;
    /// Pointer to the internal data
    std::unique_ptr<impl> m_impl;

};  // class async_copy

}  // namespace vecmem::hip
