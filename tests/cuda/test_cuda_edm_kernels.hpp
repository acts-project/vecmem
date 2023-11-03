/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "../common/edm_simple_container.hpp"
#include "vecmem/utils/types.hpp"

/// Helper function for modifying the data in a container
VECMEM_HOST_AND_DEVICE
inline void modify(unsigned int i,
                   vecmem::testing::simple_container::device& device) {

    // In the first thread modify the scalars.
    if (i == 0) {
        vecmem::testing::simple_container::count::get(device) += 2;
        vecmem::testing::simple_container::average::get(device) -= 1.0f;
    }
    // In the rest of the threads modify the vector variables.
    if (i < device.size()) {
        vecmem::testing::simple_container::measurement::get(device)[i] *= 2.0f;
        vecmem::testing::simple_container::index::get(device)[i] += 10;
    }
}

/// Helper function for filling data into a device container
VECMEM_HOST_AND_DEVICE
inline void fill(unsigned int i,
                 vecmem::testing::simple_container::device& device) {

    // In the first thread modify the scalars.
    if (i == 0) {
        vecmem::testing::simple_container::count::get(device) = 55;
        vecmem::testing::simple_container::average::get(device) = 3.141592f;
    }
    // In the rest of the threads modify the vector variables.
    if (i < device.size()) {
        vecmem::testing::simple_container::measurement::get(device)[i] =
            1.0f * static_cast<float>(i);
        vecmem::testing::simple_container::index::get(device)[i] =
            static_cast<int>(i);
    }
}

/// Modify the data in a container
void edmModify(vecmem::testing::simple_container::view view);
/// Fill data into a container
void edmFill(vecmem::testing::simple_container::view view);
