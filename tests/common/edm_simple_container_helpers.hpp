/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "edm_simple_container.hpp"

namespace vecmem::testing {

/// Fill a host container with some dummy data
void fill(simple_container::host& obj);

/// Fill a device container with some dummy data
void fill(simple_container::device& obj);

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
