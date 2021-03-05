/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cstddef>

/// Function executing a simple HIP kernel on the input/output arrays
void linearTransform( std::size_t size, const int* input, int* output );
