/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/cuda/stream_wrapper.hpp"
#include "vecmem/vecmem_cuda_export.hpp"

// System include(s).
#include <memory>

namespace vecmem::cuda {

/// Specialisation of @c vecmem::copy for CUDA
///
/// This specialisation of @c vecmem::copy, unlike @c vecmem::cuda::copy,
/// performs all of its operations asynchronously. Using the CUDA stream
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
    /// Constructor with the CUDA stream to operate on
    ///
    /// @param stream The CUDA stream to perform the copies in
    ///
    VECMEM_CUDA_EXPORT
    explicit async_copy(const stream_wrapper& stream);
    /// Constructor with a stream and flags to create CUDA events with
    ///
    /// For details on the flags, see the documentation
    /// of @c cudaEventCreateWithFlags(...) on:
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    ///
    /// @param stream The CUDA stream to perform the copies in
    /// @param event_flags Flag(s) to create internal CUDA events with
    ///
    VECMEM_CUDA_EXPORT
    async_copy(const stream_wrapper& stream, unsigned int event_flags);
    /// Move constructor
    VECMEM_CUDA_EXPORT
    async_copy(async_copy&&) noexcept;
    /// Destructor
    VECMEM_CUDA_EXPORT
    ~async_copy() noexcept override;

    /// Move assignment operator
    VECMEM_CUDA_EXPORT
    async_copy& operator=(async_copy&&) noexcept;

private:
    /// Perform an asynchronous memory copy using CUDA
    VECMEM_CUDA_EXPORT
    void do_copy(std::size_t size, const void* from, void* to,
                 type::copy_type cptype) const final;
    /// Fill a memory area using CUDA asynchronously
    VECMEM_CUDA_EXPORT
    void do_memset(std::size_t size, void* ptr, int value) const final;
    /// Create an event for synchronization
    VECMEM_CUDA_EXPORT
    event_type create_event() const final;

    /// Internal data type for the class
    struct impl;
    /// Pointer to the internal data
    std::unique_ptr<impl> m_impl;

};  // class async_copy

}  // namespace vecmem::cuda
