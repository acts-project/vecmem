/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/hip/host_memory_resource.hpp"

#include "../../utils/hip_error_handling.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

namespace vecmem::hip {

void* host_memory_resource::do_allocate(std::size_t nbytes, std::size_t) {

    void* result = nullptr;
    VECMEM_HIP_ERROR_CHECK(hipHostMalloc(&result, nbytes));
    return result;
}

void host_memory_resource::do_deallocate(void* ptr, std::size_t, std::size_t) {

    VECMEM_HIP_ERROR_CHECK(hipHostFree(ptr));
}

bool host_memory_resource::do_is_equal(
    const memory_resource& other) const noexcept {

    // Try to cast the other object to this exact type.
    const host_memory_resource* p =
        dynamic_cast<const host_memory_resource*>(&other);

    // The two are equal if they are of the same type.
    return (p != nullptr);
}

}  // namespace vecmem::hip
