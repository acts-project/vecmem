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

namespace vecmem::details {

/// Base class for implementations of the @c vecmem::memory_resource interface
///
/// This helper class is mainly meant to help with mitigating compiler warnings
/// about exporting types that inherit from standard library types. But at the
/// very least it also provides a default/conservative implementation for the
/// @c vecmem::memory_resource::do_is_equal(...) function.
///
class VECMEM_CORE_EXPORT memory_resource_base : public memory_resource {

public:
    /// Inherit the base class's constructor(s)
    using vecmem::memory_resource::memory_resource;

protected:
    /// @name Function(s) implemented from @c vecmem::memory_resource
    /// @{

    /// Compares @c *this for equality with @c other
    virtual bool do_is_equal(
        const memory_resource &other) const noexcept override;

    /// @}

};  // class memory_resource_base

}  // namespace vecmem::details
