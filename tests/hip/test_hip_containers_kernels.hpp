/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/const_device_vector_data.hpp"
#include "vecmem/containers/device_vector_data.hpp"

/// Function executing a simple HIP kernel on the input/output arrays
void linearTransform( vecmem::const_device_vector_data< int > constants,
                      vecmem::const_device_vector_data< int > input,
                      vecmem::device_vector_data< int > output );
