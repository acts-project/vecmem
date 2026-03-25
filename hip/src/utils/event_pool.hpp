/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// HIP include(s).
#include <hip/hip_runtime_api.h>

// System include(s).
#include <vector>

namespace vecmem {
namespace hip {
namespace details {

/// Pool of @c hipEvent_t objects to use in the library
///
/// It allows @c vecmem::hip::async_copy to reuse the @c hipEvent_t objects
/// that it uses for synchronization. Skipping the overhead associated with
/// creating and destroying HIP event objects.
///
/// This is based on code written by NVIDIA. Thanks to Andreas Hehn for sharing
/// that code with us!
///
class event_pool {

public:
    /// Constructor with the number of events to initially allocate
    explicit event_pool(std::size_t size = 8u);
    /// Destructor
    ~event_pool();

    /// Get an event from the pool, creating a new one if necessary
    hipEvent_t create();
    /// Return an event to the pool
    void free(hipEvent_t event);

private:
    /// The pool of events
    std::vector<hipEvent_t> m_pool;
    /// The number of currently used events
    std::size_t m_used_events = 0;

};  // class event_pool

}  // namespace details
}  // namespace hip
}  // namespace vecmem
