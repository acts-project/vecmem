/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/utils/vitis/copy.hpp"

#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cassert>
#include <string>

namespace vecmem::vitis {

/// Helper array for providing a printable name for the copy type definitions
static const std::string copy_type_printer[copy::type::count] = {
    "host to device", "device to host", "host to host", "device to device",
    "unknown"};

void copy::do_copy(std::size_t size, const void* from_ptr, void* to_ptr,
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

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(1, "called do_copy with size %lu, from %p to %p, type %s",
                     size, from_ptr, to_ptr,
                     copy_type_printer[static_cast<int>(cptype)].c_str());
}

void copy::do_memset(std::size_t size, void* ptr, int value) const {

    // Check if anything needs to be done.
    if (size == 0) {
        VECMEM_DEBUG_MSG(5, "Skipping unnecessary memory filling");
        return;
    }

    // Some sanity checks.
    assert(ptr != nullptr);

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(2, "Set %lu bytes to %i at %p with VITIS", size, value,
                     ptr);
}

}  // namespace vecmem::vitis
