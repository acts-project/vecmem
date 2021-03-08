/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/details/vector_data.hpp"

/// Perform a linear transformation using the received vectors
void linearTransform( vecmem::details::vector_data< const int > constants,
                      vecmem::details::vector_data< const int > input,
                      vecmem::details::vector_data< int > output );
