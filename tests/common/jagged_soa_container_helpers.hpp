/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "jagged_soa_container.hpp"

namespace vecmem::testing {

/// Fill a host container with some dummy data
void fill(jagged_soa_container::host& obj);

/// Fill a device container with some dummy data
void fill(jagged_soa_container::device& obj);

/// Helper function testing the equality of two containers
void compare(const jagged_soa_container::const_view& lhs,
             const jagged_soa_container::const_view& rhs);

/// Create a buffer for the tests
void make_buffer(jagged_soa_container::buffer& buffer, memory_resource& main_mr,
                 memory_resource& host_mr, data::buffer_type buffer_type);

}  // namespace vecmem::testing
