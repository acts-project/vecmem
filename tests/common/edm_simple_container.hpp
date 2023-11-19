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

namespace vecmem {
namespace testing {

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
    using count = edm::accessor<0, schema_type>;
    /// "Measurement" of something
    using measurement = edm::accessor<1, schema_type>;
    /// Global "average" of something
    using average = edm::accessor<2, schema_type>;
    /// "Index" of something
    using index = edm::accessor<3, schema_type>;

    /// @}

};  // struct simple_container

}  // namespace testing
}  // namespace vecmem
