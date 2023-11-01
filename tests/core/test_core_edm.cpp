/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/accessor.hpp"
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

typedef vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                            vecmem::edm::type::vector<float> >
    simple_schema;
typedef vecmem::edm::schema<vecmem::edm::type::scalar<const int>,
                            vecmem::edm::type::vector<const float> >
    simple_const_schema;
typedef vecmem::edm::accessor<0> simple_counts;
typedef vecmem::edm::accessor<1> simple_measurements;

TEST_F(core_edm_test, simple_view) {

    // Test the creation of simple views.
    vecmem::edm::view<simple_schema> view1;
    vecmem::edm::view<simple_const_schema> view2{view1};
}

TEST_F(core_edm_test, simple_buffer) {

    // Test the creation of fixed sized and resizable, at the same time "simple"
    // buffers.
    vecmem::edm::buffer<simple_schema> buffer1(
        10, m_resource, vecmem::data::buffer_type::fixed_size),
        buffer2(10, m_resource, vecmem::data::buffer_type::resizable);

    // "Create" views off of the buffers.
    vecmem::edm::view<simple_schema> view1{buffer1}, view2{buffer2};
    vecmem::edm::view<simple_const_schema> view3{buffer1}, view4{buffer2};

    // Check the views.
    EXPECT_NE(view1.get<0>(), nullptr);
    EXPECT_EQ(view1.get<1>().size(), 10u);
    EXPECT_EQ(view1.get<1>().capacity(), 10u);

    EXPECT_NE(view2.get<0>(), nullptr);
    EXPECT_EQ(view2.get<1>().size(), 0u);
    EXPECT_EQ(view2.get<1>().capacity(), 10u);

    EXPECT_NE(view3.get<0>(), nullptr);
    EXPECT_EQ(view3.get<1>().size(), 10u);
    EXPECT_EQ(view3.get<1>().capacity(), 10u);

    EXPECT_NE(view4.get<0>(), nullptr);
    EXPECT_EQ(view4.get<1>().size(), 0u);
    EXPECT_EQ(view4.get<1>().capacity(), 10u);
}

TEST_F(core_edm_test, simple_host) {

    // Test the creation of a simple host container.
    vecmem::edm::host<simple_schema> host1{m_resource};

    // Make views out of it.
    vecmem::edm::view<simple_schema> ncview1 = vecmem::get_data(host1);
    vecmem::edm::view<simple_const_schema> cview1 = [](const auto& host) {
        return vecmem::get_data(host);
    }(host1);

    // Make trivial checks on the contents of the view(s).
    EXPECT_EQ(ncview1.get<0>(), host1.get<0>().get());
    EXPECT_EQ(cview1.get<1>().size(), host1.get<1>().size());

    // Fill the host container with some data.
    simple_counts::get(host1) = 10;
    simple_measurements::get(host1).push_back(1.0f);

    // Check the contents of the host container.
    const auto& host1c = host1;
    EXPECT_EQ(simple_counts::get(host1c), 10);
    EXPECT_EQ(simple_measurements::get(host1c).size(), 1u);
    EXPECT_FLOAT_EQ(simple_measurements::get(host1c)[0], 1.0f);
}
