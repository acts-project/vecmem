/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/vecmem_sycl_export.hpp"

// System include(s).
#include <memory>

namespace vecmem::sycl {

// Forward declaration(s).
namespace details {
class opaque_queue;
}

// Disable the warning(s) about inheriting from/using standard library types
// with an exported class.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif  // MSVC

/// Wrapper class for @c cl::sycl::queue
///
/// It is necessary for passing around SYCL queue objects in code that should
/// not be directly exposed to the SYCL headers.
///
class VECMEM_SYCL_EXPORT queue_wrapper {

public:
    /// Construct a queue for the default device
    queue_wrapper();
    /// Wrap an existing @c cl::sycl::queue object
    ///
    /// Without taking ownership of it!
    ///
    queue_wrapper(void* queue);

    /// Copy constructor
    queue_wrapper(const queue_wrapper& parent);
    /// Move constructor
    queue_wrapper(queue_wrapper&& parent);

    /// Destructor
    ///
    /// The destructor is declared and implemented explicitly as an empty
    /// function to make sure that client code would not try to generate it
    /// itself. Leading to problems about the symbols of
    /// @c vecmem::sycl::details::opaque_queue not being available in
    /// client code.
    ///
    ~queue_wrapper();

    /// Copy assignment
    queue_wrapper& operator=(const queue_wrapper& rhs);
    /// Move assignment
    queue_wrapper& operator=(queue_wrapper&& rhs);

    /// Access a typeless pointer to the managed @c cl::sycl::queue object
    void* queue();
    /// Access a typeless pointer to the managed @c cl::sycl::queue object
    const void* queue() const;

private:
    /// Bare pointer to the wrapped @c cl::sycl::queue object
    void* m_queue;
    /// Smart pointer to the managed @c cl::sycl::queue object
    std::unique_ptr<details::opaque_queue> m_managedQueue;

};  // class queue_wrapper

}  // namespace vecmem::sycl

// Re-enable the warning(s).
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC
