/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/vecmem_sycl_export.hpp"

// System include(s).
#include <memory>
#include <string>

namespace vecmem::sycl {

/// Wrapper class for @c ::sycl::queue
///
/// It is necessary for passing around SYCL queue objects in code that should
/// not be directly exposed to the SYCL headers.
///
class queue_wrapper {

public:
    /// Construct a queue for the default device
    VECMEM_SYCL_EXPORT
    queue_wrapper();
    /// Wrap an existing @c ::sycl::queue object
    ///
    /// Without taking ownership of it!
    ///
    VECMEM_SYCL_EXPORT
    queue_wrapper(void* queue);

    /// Copy constructor
    VECMEM_SYCL_EXPORT
    queue_wrapper(const queue_wrapper& parent);
    /// Move constructor
    VECMEM_SYCL_EXPORT
    queue_wrapper(queue_wrapper&& parent) noexcept;

    /// Destructor
    VECMEM_SYCL_EXPORT
    ~queue_wrapper();

    /// Copy assignment
    VECMEM_SYCL_EXPORT
    queue_wrapper& operator=(const queue_wrapper& rhs);
    /// Move assignment
    VECMEM_SYCL_EXPORT
    queue_wrapper& operator=(queue_wrapper&& rhs) noexcept;

    /// Access a typeless pointer to the managed @c ::sycl::queue object
    VECMEM_SYCL_EXPORT
    void* queue();
    /// Access a typeless pointer to the managed @c ::sycl::queue object
    VECMEM_SYCL_EXPORT
    const void* queue() const;

    /// Wait for all tasks in the queue to complete
    VECMEM_SYCL_EXPORT
    void synchronize();

    /// Get the name of the device that the queue operates on
    VECMEM_SYCL_EXPORT
    std::string device_name() const;

    /// Check if it's a CPU queue
    VECMEM_SYCL_EXPORT
    bool is_cpu() const;
    /// Check if it's a GPU queue
    VECMEM_SYCL_EXPORT
    bool is_gpu() const;
    /// Check if it's an accelerator (FPGA) queue
    VECMEM_SYCL_EXPORT
    bool is_accelerator() const;

    /// Check if it's an OpenCL queue
    VECMEM_SYCL_EXPORT
    bool is_opencl() const;
    /// Check if it's a Level-0 queue
    VECMEM_SYCL_EXPORT
    bool is_level0() const;
    /// Check if it's a CUDA queue
    VECMEM_SYCL_EXPORT
    bool is_cuda() const;
    /// Check if it's a HIP queue
    VECMEM_SYCL_EXPORT
    bool is_hip() const;

private:
    /// Structure holding the internals of the class
    struct impl;
    /// Pointer to the internal structure
    std::unique_ptr<impl> m_impl;

};  // class queue_wrapper

}  // namespace vecmem::sycl
