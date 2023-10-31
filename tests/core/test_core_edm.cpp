/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/buffer.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test case for the EDM code
class core_edm_test : public testing::Test {

protected:
    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;
    /// Helper object for the memory copies.
    vecmem::copy m_copy;

};  // class core_device_container_test

TEST_F(core_edm_test, simple_buffer) {

    // Test the creation of fixed sized and resizable, at the same time "simple"
    // buffers.
    vecmem::edm::buffer<vecmem::edm::type::scalar<int>,
                        vecmem::edm::type::vector<float> >
        buffer1(10, m_resource, vecmem::data::buffer_type::fixed_size),
        buffer2(10, m_resource, vecmem::data::buffer_type::resizable);
}
