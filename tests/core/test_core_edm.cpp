/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/accessor.hpp"
#include "vecmem/edm/buffer.hpp"
#include "vecmem/edm/device.hpp"
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

/// A "simple EDM" for the tests
namespace simple_edm {

/// Layout schema for this type
using schema = vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                                   vecmem::edm::type::vector<float> >;
/// Constant version of the schema
using const_schema = vecmem::edm::add_const<schema>::type;

/// Global "counts" for the container
static const vecmem::edm::accessor<0> counts;
/// "Measurements" for the container
static const vecmem::edm::accessor<1> measurements;

/// Host container type
using host = vecmem::edm::host<schema>;

/// (Non-const) Device container type
using device = vecmem::edm::device<schema>;
/// (Const) Device container type
using const_device = vecmem::edm::device<const_schema>;

/// (Non-const) View type
using view = vecmem::edm::view<schema>;
/// (Const) view type
using const_view = vecmem::edm::view<const_schema>;

/// Buffer type
using buffer = vecmem::edm::buffer<schema>;

}  // namespace simple_edm

TEST_F(core_edm_test, simple_view) {

    // Test the creation of simple views.
    simple_edm::view view1;
    simple_edm::const_view view2{view1};
}

TEST_F(core_edm_test, simple_buffer) {

    // Test the creation of fixed sized and resizable, at the same time "simple"
    // buffers.
    simple_edm::buffer buffer1(10, m_resource,
                               vecmem::data::buffer_type::fixed_size);
    simple_edm::buffer buffer2(10, m_resource,
                               vecmem::data::buffer_type::resizable);

    // "Create" views off of the buffers.
    simple_edm::view view1{buffer1}, view2{buffer2};
    simple_edm::const_view view3{buffer1}, view4{buffer2};

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

    // Create device containers from the views.
    simple_edm::device device1{view1}, device2{view2};
    simple_edm::const_device device3{view1}, device4{view2}, device5{view3},
        device6{view4};

    // Set some data on the device containers.
    simple_edm::counts(device1) = 10;
    for (int i = 0; i < 10; ++i) {
        simple_edm::measurements(device1)[i] = static_cast<float>(i);
    }
    simple_edm::counts(device2) = 5;
    for (int i = 0; i < 5; ++i) {
        simple_edm::measurements(device2).push_back(2.f *
                                                    static_cast<float>(i));
    }

    // Check the device containers.
    EXPECT_EQ(simple_edm::counts(device1), 10);
    EXPECT_EQ(simple_edm::measurements(device1).size(), 10u);
    EXPECT_EQ(simple_edm::measurements(device1).capacity(), 10u);
    EXPECT_EQ(simple_edm::counts(device3), 10);
    EXPECT_EQ(simple_edm::measurements(device3).size(), 10u);
    EXPECT_EQ(simple_edm::measurements(device3).capacity(), 10u);
    EXPECT_EQ(simple_edm::counts(device5), 10);
    EXPECT_EQ(simple_edm::measurements(device5).size(), 10u);
    EXPECT_EQ(simple_edm::measurements(device5).capacity(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(simple_edm::measurements(device1)[i],
                        static_cast<float>(i));
        EXPECT_FLOAT_EQ(simple_edm::measurements(device3)[i],
                        static_cast<float>(i));
        EXPECT_FLOAT_EQ(simple_edm::measurements(device5)[i],
                        static_cast<float>(i));
    }
    EXPECT_EQ(simple_edm::counts(device2), 5);
    EXPECT_EQ(simple_edm::measurements(device2).size(), 5u);
    EXPECT_EQ(simple_edm::measurements(device2).capacity(), 10u);
    EXPECT_EQ(simple_edm::counts(device4), 5);
    EXPECT_EQ(simple_edm::measurements(device4).size(), 5u);
    EXPECT_EQ(simple_edm::measurements(device4).capacity(), 10u);
    EXPECT_EQ(simple_edm::counts(device6), 5);
    EXPECT_EQ(simple_edm::measurements(device6).size(), 5u);
    EXPECT_EQ(simple_edm::measurements(device6).capacity(), 10u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(simple_edm::measurements(device2)[i],
                        2.f * static_cast<float>(i));
        EXPECT_FLOAT_EQ(simple_edm::measurements(device4)[i],
                        2.f * static_cast<float>(i));
        EXPECT_FLOAT_EQ(simple_edm::measurements(device6)[i],
                        2.f * static_cast<float>(i));
    }
}

TEST_F(core_edm_test, simple_host) {

    // Test the creation of a simple host container.
    simple_edm::host host1{m_resource};

    // Fill the host container with some data.
    simple_edm::counts(host1) = 10;
    simple_edm::measurements(host1).push_back(1.0f);

    // Check the contents of the host container.
    const auto& host1c = host1;
    EXPECT_EQ(simple_edm::counts(host1c), 10);
    EXPECT_EQ(simple_edm::measurements(host1c).size(), 1u);
    EXPECT_FLOAT_EQ(simple_edm::measurements(host1c)[0], 1.0f);

    // Make views out of it.
    simple_edm::view ncview1 = vecmem::get_data(host1);
    simple_edm::const_view cview1 = [](const auto& host) {
        return vecmem::get_data(host);
    }(host1);

    // Make trivial checks on the contents of the view(s).
    EXPECT_EQ(ncview1.get<0>(), host1.get<0>().get());
    EXPECT_EQ(cview1.get<1>().size(), host1.get<1>().size());

    // Make device containers out of the views.
    simple_edm::device device1{ncview1};
    simple_edm::const_device device2{ncview1}, device3{cview1};

    // Check the contents of the device containers.
    EXPECT_EQ(simple_edm::counts(device1), 10);
    EXPECT_EQ(simple_edm::counts(device2), 10);
    EXPECT_EQ(simple_edm::counts(device3), 10);
    EXPECT_EQ(simple_edm::measurements(device1).size(), 1u);
    EXPECT_EQ(simple_edm::measurements(device2).size(), 1u);
    EXPECT_EQ(simple_edm::measurements(device3).size(), 1u);
    EXPECT_FLOAT_EQ(simple_edm::measurements(device1)[0], 1.0f);
    EXPECT_FLOAT_EQ(simple_edm::measurements(device2)[0], 1.0f);
    EXPECT_FLOAT_EQ(simple_edm::measurements(device3)[0], 1.0f);
}
