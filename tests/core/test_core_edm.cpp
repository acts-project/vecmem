/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/buffer.hpp"
#include "vecmem/edm/host.hpp"
#include "vecmem/edm/view.hpp"
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

TEST_F(core_edm_test, simple_view) {

    // Test the creation of a simple view.
    vecmem::edm::view<vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                                          vecmem::edm::type::vector<float> > >
        view1;

    vecmem::edm::view<
        vecmem::edm::schema<vecmem::edm::type::scalar<const int>,
                            vecmem::edm::type::vector<const float> > >
        view2{view1};
}

TEST_F(core_edm_test, simple_buffer) {

    // Test the creation of fixed sized and resizable, at the same time "simple"
    // buffers.
    vecmem::edm::buffer<vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                                            vecmem::edm::type::vector<float> > >
        buffer1(10, m_resource, vecmem::data::buffer_type::fixed_size),
        buffer2(10, m_resource, vecmem::data::buffer_type::resizable);

    // "Create" views off of the buffers.
    vecmem::edm::view<vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                                          vecmem::edm::type::vector<float> > >
        view1{buffer1}, view2{buffer2};
    vecmem::edm::view<
        vecmem::edm::schema<vecmem::edm::type::scalar<const int>,
                            vecmem::edm::type::vector<const float> > >
        view3{buffer1}, view4{buffer2};
}

TEST_F(core_edm_test, simple_host) {

    // Test the creation of a simple host container.
    vecmem::edm::host<vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                                          vecmem::edm::type::vector<float> > >
        host1;

    // Make a (non-const) view out of it.
    auto view1 = vecmem::get_data(host1);
}
