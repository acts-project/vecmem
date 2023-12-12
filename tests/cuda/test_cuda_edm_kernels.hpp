/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "../common/jagged_soa_container.hpp"
#include "../common/simple_soa_container.hpp"

/// Fill a simple SoA container with some data.
void cudaSimpleFill(vecmem::testing::simple_soa_container::view view);

/// Fill a jagged SoA container with some data.
void cudaJaggedFill(vecmem::testing::jagged_soa_container::view view);

/// Modify data in a simple SoA container.
void cudaSimpleModify(vecmem::testing::simple_soa_container::view view);

/// Modify data in a jagged SoA container.
void cudaJaggedModify(vecmem::testing::jagged_soa_container::view view);
