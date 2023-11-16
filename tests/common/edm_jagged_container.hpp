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
struct jagged_container
    : public edm::container<
          edm::type::scalar<int>, edm::type::vector<float>,
          edm::type::jagged_vector<double>, edm::type::scalar<float>,
          edm::type::jagged_vector<int>, edm::type::vector<int> > {

    /// @name Accessors to the individual variables in the collection
    /// @{

    /// Global "count" of something
    using count = edm::accessor<0, schema>;
    /// "Measurement" of something
    using measurement = edm::accessor<1, schema>;
    /// "Measurements" of something
    using measurements = edm::accessor<2, schema>;
    /// Global "average" of something
    using average = edm::accessor<3, schema>;
    /// "Indices" of something
    using indices = edm::accessor<4, schema>;
    /// "Index" of something
    using index = edm::accessor<5, schema>;

    /// @}

};  // struct jagged_container

}  // namespace testing
}  // namespace vecmem
