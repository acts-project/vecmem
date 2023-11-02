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
    : public edm::container<
          edm::type::scalar<int>, vecmem::edm::type::vector<float>,
          vecmem::edm::type::scalar<float>, vecmem::edm::type::vector<int> > {

    /// @name Accessors to the individual variables in the collection
    /// @{

    /// Global "count" of something
    using count = edm::accessor<0>;
    /// "Measurement" of something
    using measurement = edm::accessor<1>;
    /// Global "average" of something
    using average = edm::accessor<2>;
    /// "Index" of something
    using index = edm::accessor<3>;

    /// @}

};  // struct simple_container

}  // namespace vecmem::testing
