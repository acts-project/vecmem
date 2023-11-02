/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/accessor.hpp"
#include "vecmem/edm/container.hpp"

namespace vecmem::testing {

/// "Simple" container for the tests
///
/// Meaning that it would not have any jagged vector variables in it...
///
struct simple_container
    : public edm::container<edm::type::scalar<int>, edm::type::vector<float>,
                            edm::type::scalar<float>, edm::type::vector<int> > {

    /// @name Accessors to the individual variables in the collection
    /// @{

    /// Global "count" of something
    using count = edm::accessor<0, schema>;
    /// "Measurement" of something
    using measurement = edm::accessor<1, schema>;
    /// Global "average" of something
    using average = edm::accessor<2, schema>;
    /// "Index" of something
    using index = edm::accessor<3, schema>;

    /// @}

};  // struct simple_container

/// Helper function testing the equality of two host containers
void compare(const simple_container::host& lhs,
             const simple_container::host& rhs);

/// Helper function testing the equality of two device containers
void compare(const simple_container::const_device& lhs,
             const simple_container::const_device& rhs);

/// Helper function testing the equality of a host and a device container
void compare(const simple_container::host& lhs,
             const simple_container::const_device& rhs);

}  // namespace vecmem::testing
