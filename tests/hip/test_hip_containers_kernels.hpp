/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/details/vector_view.hpp"

/// Function executing a simple HIP kernel on the input/output arrays
void linearTransform( vecmem::details::vector_view< const int > constants,
                      vecmem::details::vector_view< const int > input,
                      vecmem::details::vector_view< int > output );
