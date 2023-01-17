/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/vecmem_cuda_export.hpp"

// System include(s).
#include <memory>
#include <string>

namespace vecmem {
namespace cuda {

/// @brief Namespace for types that should not be used directly by clients
namespace details {
class opaque_stream;
}

// Disable the warning(s) about inheriting from/using standard library types
// with an exported class.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif  // MSVC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 1394
#endif  // CUDA disgnostics

/// Wrapper class for @c cudaStream_t
///
/// It is necessary for passing around CUDA stream objects in code that should
/// not be directly exposed to the CUDA header(s).
///
class VECMEM_CUDA_EXPORT stream_wrapper {

public:
    /// Invalid/default device identifier
    static constexpr int INVALID_DEVICE = -1;

    /// Construct a new stream (for the specified device)
    stream_wrapper(int device = INVALID_DEVICE);
    /// Wrap an existing @c cudaStream_t object
    ///
    /// Without taking ownership of it!
    ///
    stream_wrapper(void* stream);

    /// Copy constructor
    stream_wrapper(const stream_wrapper& parent);
    /// Move constructor
    stream_wrapper(stream_wrapper&& parent);

    /// Destructor
    ~stream_wrapper();

    /// Copy assignment
    stream_wrapper& operator=(const stream_wrapper& rhs);
    /// Move assignment
    stream_wrapper& operator=(stream_wrapper&& rhs);

    /// Access a typeless pointer to the managed @c cudaStream_t object
    void* stream() const;

    /// Wait for all queued tasks from the stream to complete
    void synchronize();

private:
    /// Bare pointer to the wrapped @c cudaStream_t object
    void* m_stream;
    /// Smart pointer to the managed @c cudaStream_t object
    std::shared_ptr<details::opaque_stream> m_managedStream;

};  // class stream_wrapper

}  // namespace cuda
}  // namespace vecmem

// Re-enable the warning(s).
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diagnostic pop
#endif  // CUDA disgnostics
