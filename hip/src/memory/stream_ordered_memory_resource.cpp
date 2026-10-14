/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/hip/stream_ordered_memory_resource.hpp"

#include "../utils/get_stream.hpp"
#include "../utils/hip_error_handling.hpp"
#include "vecmem/utils/debug.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

// System include(s).
#include <cassert>
#include <stdexcept>

namespace vecmem::hip {

stream_ordered_memory_resource::stream_ordered_memory_resource(
    const stream_wrapper& stream)
    : m_stream(stream) {}

stream_ordered_memory_resource::~stream_ordered_memory_resource() noexcept =
    default;

void* stream_ordered_memory_resource::do_allocate(std::size_t bytes,
                                                  std::size_t) {

    if (bytes == 0) {
        throw std::bad_alloc();
    }

    // Allocate the memory.
    void* res = nullptr;
    VECMEM_HIP_ERROR_CHECK(
        hipMallocAsync(&res, bytes, details::get_stream(m_stream)));
    VECMEM_DEBUG_MSG(2, "Allocated %ld bytes at %p on stream %p", bytes, res,
                     m_stream.stream());
    return res;
}

void stream_ordered_memory_resource::do_deallocate(void* p, std::size_t bytes,
                                                   std::size_t) {

    assert(p != nullptr);
    if (bytes == 0u) {
        return;
    }

    // Free the memory.
    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p on stream %p", p,
                     m_stream.stream());
    VECMEM_HIP_ERROR_CHECK(hipFreeAsync(p, details::get_stream(m_stream)));
}

bool stream_ordered_memory_resource::do_is_equal(
    const memory_resource& other) const noexcept {

    // Try to cast the other object to this exact type.
    auto p = dynamic_cast<const stream_ordered_memory_resource*>(&other);

    // The two are equal if they operate on the same stream.
    return ((p != nullptr) && (p->m_stream.stream() == m_stream.stream()));
}

}  // namespace vecmem::hip
