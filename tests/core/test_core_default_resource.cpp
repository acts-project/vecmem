/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

TEST(core_default_resource_test, vector_default_resource) {
    vecmem::vector<float> v;
    v.emplace_back(1);
    v.emplace_back(3);
    v.emplace_back(2);
    EXPECT_EQ(v.back(), 2);
}
