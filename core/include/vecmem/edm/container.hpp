/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/buffer.hpp"
#include "vecmem/edm/device.hpp"
#include "vecmem/edm/host.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/edm/view.hpp"

/// Namespace for the types serving as base classes for an SoA event data model
namespace vecmem::edm {

/// Description of a container, with all of the types needed to use it
///
/// @tparam ...VARIABLES The variable types stored in the container
///
template <typename... VARIABLES>
struct container {

    /// Schema for this container
    using schema = vecmem::edm::schema<VARIABLES...>;
    /// Constant version of the schema
    using const_schema = typename vecmem::edm::add_const<schema>::type;

    /// Host container type
    using host = vecmem::edm::host<schema>;

    /// (Non-const) Device container type
    using device = vecmem::edm::device<schema>;
    /// (Const) Device container type
    using const_device = vecmem::edm::device<const_schema>;

    /// (Non-const) View type
    using view = vecmem::edm::view<schema>;
    /// (Const) view type
    using const_view = vecmem::edm::view<const_schema>;

    /// Buffer type
    using buffer = vecmem::edm::buffer<schema>;

};  // struct container

}  // namespace vecmem::edm
