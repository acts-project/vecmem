/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <cstddef>

namespace vecmem::details {

/// Struct used for deallocating an allocated memory block
///
/// It can be used to make things like @c std::unique_ptr talk to
/// @c vecmem::memory_resource. Note however that this is *not* a "deleter"
/// type. It *does not* call any custom destructors. It merely de-allocates
/// a memory block.
///
class VECMEM_CORE_EXPORT deallocator {

public:
    /// Constructor
    deallocator(std::size_t bytes, memory_resource& resource);

    /// Copy constructor
    deallocator(const deallocator&) = default;
    /// Move constructor
    deallocator(deallocator&&) = default;
    /// Copy assignment operator
    deallocator& operator=(const deallocator&) = default;
    /// Move assignment operator
    deallocator& operator=(deallocator&&) = default;

    /// Operator performing the deletion of the object.
    void operator()(void* ptr);

private:
    /// The number of bytes allocated for the memory block
    std::size_t m_bytes;
    /// The memory resource used for deleting the memory block
    memory_resource* m_resource;

};  // struct deallocator

}  // namespace vecmem::details
