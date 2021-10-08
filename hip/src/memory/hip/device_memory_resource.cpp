/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/hip/device_memory_resource.hpp"

#include "../../utils/get_device.hpp"
#include "../../utils/hip_error_handling.hpp"
#include "../../utils/run_on_device.hpp"
#include "vecmem/utils/debug.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

namespace vecmem::hip {

device_memory_resource::device_memory_resource(int device)
    : m_device(device == INVALID_DEVICE ? details::get_device() : device) {}

void* device_memory_resource::do_allocate(std::size_t nbytes, std::size_t) {

    // Allocate the memory.
    void* result = nullptr;
    (details::run_on_device(m_device))([&result, nbytes]() {
        VECMEM_HIP_ERROR_CHECK(hipMalloc(&result, nbytes));
    });
    VECMEM_DEBUG_MSG(4, "Allocated %ld bytes at %p on device %i", nbytes,
                     result, m_device);
    return result;
}

void device_memory_resource::do_deallocate(void* ptr, std::size_t,
                                           std::size_t) {

    // Free the memory.
    VECMEM_DEBUG_MSG(4, "De-allocating memory at %p on device %i", ptr,
                     m_device);
    (details::run_on_device(m_device))(
        [ptr]() { VECMEM_HIP_ERROR_CHECK(hipFree(ptr)); });
}

bool device_memory_resource::do_is_equal(
    const memory_resource& other) const noexcept {

    // Try to cast the other object to this exact type.
    const device_memory_resource* p =
        dynamic_cast<const device_memory_resource*>(&other);

    // The two are equal if they operate on the same device.
    return ((p != nullptr) && (p->m_device == m_device));
}

}  // namespace vecmem::hip
