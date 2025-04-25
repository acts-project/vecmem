/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/utils/cuda/async_copy.hpp"

#include "../cuda_error_handling.hpp"
#include "../cuda_wrappers.hpp"
#include "vecmem/utils/debug.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <array>
#include <cassert>
#include <exception>
#include <string>

namespace {

/// CUDA specific implementation of the abstract event interface
struct cuda_event : public vecmem::abstract_event {

    /// Constructor with the created event.
    explicit cuda_event(cudaEvent_t event) : m_event(event) {
        assert(m_event != nullptr);
    }
    /// Copy constructor
    cuda_event(const cuda_event&) = delete;
    /// Move constructor
    cuda_event(cuda_event&& parent) noexcept : m_event(parent.m_event) {
        parent.m_event = nullptr;
    }
    /// Destructor
    ~cuda_event() override {
        // Check if the user forgot to wait on this asynchronous event.
        if (m_event != nullptr) {
            // If so, wait implicitly now.
            VECMEM_DEBUG_MSG(1, "Asynchronous CUDA event was not waited on!");
            cuda_event::wait();
#ifdef VECMEM_FAIL_ON_ASYNC_ERRORS
            // If the user wants to fail on asynchronous errors, do so now.
            std::terminate();
#endif  // VECMEM_FAIL_ON_ASYNC_ERRORS
        }
    }

    /// Copy assignment
    cuda_event& operator=(const cuda_event&) = delete;
    /// Move assignment
    cuda_event& operator=(cuda_event&& rhs) noexcept {
        if (this != &rhs) {
            m_event = rhs.m_event;
            rhs.m_event = nullptr;
        }
        return *this;
    }

    /// Synchronize on the underlying CUDA event
    void wait() override {
        if (m_event == nullptr) {
            return;
        }
        VECMEM_CUDA_ERROR_CHECK(cudaEventSynchronize(m_event));
        cuda_event::ignore();
    }

    /// Ignore the underlying CUDA event
    void ignore() override {
        if (m_event == nullptr) {
            return;
        }
        VECMEM_CUDA_ERROR_CHECK(cudaEventDestroy(m_event));
        m_event = nullptr;
    }

    /// The CUDA event wrapped by this struct
    cudaEvent_t m_event;

};  // struct cuda_event

}  // namespace

namespace vecmem::cuda {

/// Helper array for translating between the vecmem and CUDA copy type
/// definitions
static constexpr std::array<cudaMemcpyKind, copy::type::count>
    copy_type_translator = {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                            cudaMemcpyHostToHost, cudaMemcpyDeviceToDevice,
                            cudaMemcpyDefault};

/// Helper array for providing a printable name for the copy type
/// definitions
static const std::array<std::string, copy::type::count> copy_type_printer = {
    "host to device", "device to host", "host to host", "device to device",
    "unknown"};

async_copy::async_copy(const stream_wrapper& stream) : m_stream(stream) {}

async_copy::~async_copy() noexcept = default;

void async_copy::do_copy(std::size_t size, const void* from_ptr, void* to_ptr,
                         type::copy_type cptype) const {

    // Check if anything needs to be done.
    if (size == 0) {
        VECMEM_DEBUG_MSG(5, "Skipping unnecessary memory copy");
        return;
    }

    // Some sanity checks.
    assert(from_ptr != nullptr);
    assert(to_ptr != nullptr);
    assert(static_cast<int>(cptype) >= 0);
    assert(static_cast<int>(cptype) < static_cast<int>(copy::type::count));

    // Perform the copy.
    VECMEM_CUDA_ERROR_CHECK(cudaMemcpyAsync(to_ptr, from_ptr, size,
                                            copy_type_translator[cptype],
                                            details::get_stream(m_stream)));

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(1,
                     "Initiated asynchronous %s memory copy of %lu bytes "
                     "from %p to %p",
                     copy_type_printer[cptype].c_str(), size, from_ptr, to_ptr);
}

void async_copy::do_memset(std::size_t size, void* ptr, int value) const {

    // Check if anything needs to be done.
    if (size == 0) {
        VECMEM_DEBUG_MSG(5, "Skipping unnecessary memory filling");
        return;
    }

    // Some sanity checks.
    assert(ptr != nullptr);

    // Perform the operation.
    VECMEM_CUDA_ERROR_CHECK(
        cudaMemsetAsync(ptr, value, size, details::get_stream(m_stream)));

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(
        2, "Initiated setting %lu bytes to %i at %p asynchronously with CUDA",
        size, value, ptr);
}

async_copy::event_type async_copy::create_event() const {

    // Create a CUDA event.
    cudaEvent_t cudaEvent = nullptr;
    VECMEM_CUDA_ERROR_CHECK(cudaEventCreate(&cudaEvent));

    // Create a smart pointer around it to make memory management a little
    // safer.
    auto event = std::make_unique<::cuda_event>(cudaEvent);

    // Record it into the copy object's CUDA stream.
    VECMEM_CUDA_ERROR_CHECK(
        cudaEventRecord(cudaEvent, details::get_stream(m_stream)));

    // Return the smart pointer.
    return event;
}

}  // namespace vecmem::cuda
