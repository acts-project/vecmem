/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/utils/copy.hpp"

#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cstring>

namespace vecmem {

void copy::do_copy(std::size_t size, const void* from_ptr, void* to_ptr,
                   type::copy_type) {

    // Perform a simple POSIX memory copy.
    ::memcpy(to_ptr, from_ptr, size);

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(4,
                     "Performed POSIX memory copy of %lu bytes from %p "
                     "to %p",
                     size, from_ptr, to_ptr);
}

void copy::do_memset(std::size_t size, void* ptr, int value) {

    // Perform the POSIX memory setting operation.
    ::memset(ptr, value, size);

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(4, "Set %lu bytes to %i at %p with POSIX memset", size,
                     value, ptr);
}

}  // namespace vecmem
