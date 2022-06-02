/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/cuda/stream_wrapper.hpp"
#include "vecmem/vecmem_cuda_export.hpp"

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
class VECMEM_CUDA_EXPORT async_copy : public vecmem::copy {

public:
    /// Constructor with the stream to operate on
    async_copy(const stream_wrapper& stream);

    /// Virtual destructor
    virtual ~async_copy() {}

protected:
    /// Perform an asynchronous memory copy using CUDA
    virtual void do_copy(std::size_t size, const void* from, void* to,
                         type::copy_type cptype) override;
    /// Fill a memory area using CUDA asynchronously
    virtual void do_memset(std::size_t size, void* ptr, int value) override;

private:
    /// The stream that the copies are performed on
    stream_wrapper m_stream;

};  // class async_copy

}  // namespace vecmem::cuda
