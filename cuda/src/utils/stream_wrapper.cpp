/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/cuda/stream_wrapper.hpp"

#include "cuda_error_handling.hpp"
#include "cuda_wrappers.hpp"
#include "get_device_name.hpp"
#include "select_device.hpp"

// VecMem include(s).
#include "vecmem/utils/debug.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <cassert>

namespace vecmem::cuda {
namespace details {

/// Struct responsible for managing the lifetime of a CUDA stream
struct stream_owner {

    /// Constructor
    stream_owner() : m_stream(nullptr) {
        VECMEM_CUDA_ERROR_CHECK(cudaStreamCreate(&m_stream));
    }
    /// Destructor
    ~stream_owner() {
        // Don't check the return value of the stream destruction. This is
        // because if the holder of this stream is only destroyed during the
        // termination of the application in which it was created, the CUDA
        // runtime may have already deleted all streams by the time that this
        // function would try to delete it.
        //
        // This is not the most robust thing ever, but detecting reliably when
        // this destructor is executed as part of the final operations of an
        // application, would be too platform specific and fragile of an
        // operation.
        cudaStreamDestroy(m_stream);
    }

    /// The managed stream
    cudaStream_t m_stream;

};  // struct stream_owner

}  // namespace details

struct stream_wrapper::impl {
    /// Bare pointer to the wrapped @c cudaStream_t object
    cudaStream_t m_stream = nullptr;
    /// Smart pointer to the managed @c cudaStream_t object
    std::shared_ptr<details::stream_owner> m_managedStream;
    /// Device identifier for the stream
    int m_device = INVALID_DEVICE;
};

stream_wrapper::stream_wrapper(int device) : m_impl{std::make_unique<impl>()} {

    // Make sure that the stream is constructed on the correct device.
    details::select_device dev_selector(
        device == INVALID_DEVICE ? details::get_device() : device);

    // Construct the stream.
    m_impl->m_managedStream = std::make_shared<details::stream_owner>();
    m_impl->m_stream = m_impl->m_managedStream->m_stream;
    m_impl->m_device = dev_selector.device();

    // Tell the user what happened.
    VECMEM_DEBUG_MSG(1, "Created stream on device: %s",
                     details::get_device_name(dev_selector.device()).c_str());
}

stream_wrapper::stream_wrapper(void* stream)
    : m_impl{std::make_unique<impl>()} {

    assert(stream != nullptr);
    m_impl->m_stream = static_cast<cudaStream_t>(stream);
}

stream_wrapper::stream_wrapper(const stream_wrapper& parent)
    : m_impl{std::make_unique<impl>()} {

    *m_impl = *(parent.m_impl);
}

stream_wrapper::stream_wrapper(stream_wrapper&& parent) = default;

stream_wrapper::~stream_wrapper() = default;

stream_wrapper& stream_wrapper::operator=(const stream_wrapper& rhs) {

    if (this != &rhs) {
        *m_impl = *(rhs.m_impl);
    }
    return *this;
}

stream_wrapper& stream_wrapper::operator=(stream_wrapper&& rhs) = default;

void* stream_wrapper::stream() const {

    return m_impl->m_stream;
}

void stream_wrapper::synchronize() {

    assert(m_impl->m_stream != nullptr);
    VECMEM_CUDA_ERROR_CHECK(cudaStreamSynchronize(m_impl->m_stream));
}

std::string stream_wrapper::device_name() const {

    if (m_impl->m_device == INVALID_DEVICE) {
        return "Unknown";
    } else {
        return details::get_device_name(m_impl->m_device);
    }
}

}  // namespace vecmem::cuda
