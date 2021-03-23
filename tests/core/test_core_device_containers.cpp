/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <cstring>
#include <type_traits>

/// Test case for the custom device container types
class core_device_container_test : public testing::Test {

}; // class core_device_container_test

/// Test that the "simple" data types are trivially constructible.
TEST_F( core_device_container_test, trivial_construct ) {

   EXPECT_TRUE( std::is_trivially_default_constructible<
                   vecmem::data::vector_view< const int > >() );
   EXPECT_TRUE( std::is_trivially_constructible<
                   vecmem::data::vector_view< const int > >() );
   EXPECT_TRUE( std::is_trivially_copy_constructible<
                   vecmem::data::vector_view< const int > >() );

   EXPECT_TRUE( std::is_trivially_default_constructible<
                   vecmem::data::vector_view< int > >() );
   EXPECT_TRUE( std::is_trivially_constructible<
                   vecmem::data::vector_view< int > >() );
   EXPECT_TRUE( std::is_trivially_copy_constructible<
                   vecmem::data::vector_view< int > >() );
}

/// Test(s) for @c vecmem::data::vector_buffer
TEST_F( core_device_container_test, vector_buffer ) {

   // Create a dummy vector in regular host memory.
   std::vector< int > host_vector { 1, 2, 3, 4, 5 };
   auto host_data = vecmem::get_data( host_vector );

   // Create an "owning copy" of the host vector's memory.
   vecmem::host_memory_resource resource;
   vecmem::data::vector_buffer< int >
      device_data( host_data.m_size, resource );
   memcpy( device_data.m_ptr, host_data.m_ptr,
           host_data.m_size * sizeof( int ) );

   // Do some basic tests.
   EXPECT_EQ( device_data.m_size, host_vector.size() );
   EXPECT_EQ( memcmp( host_data.m_ptr, device_data.m_ptr,
                      host_data.m_size * sizeof( int ) ), 0 );
}
