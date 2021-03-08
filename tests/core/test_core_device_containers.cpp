/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/details/vector_data.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <type_traits>

/// Test case for the custom device container types
class core_device_container_test : public testing::Test {

}; // class core_device_container_test

// Test that the "simple" data types are trivially constructible.
TEST_F( core_device_container_test, trivial_construct ) {

   EXPECT_TRUE( std::is_trivially_default_constructible<
                   vecmem::details::vector_data< const int > >() );
   EXPECT_TRUE( std::is_trivially_constructible<
                   vecmem::details::vector_data< const int > >() );
   EXPECT_TRUE( std::is_trivially_copy_constructible<
                   vecmem::details::vector_data< const int > >() );

   EXPECT_TRUE( std::is_trivially_default_constructible<
                   vecmem::details::vector_data< int > >() );
   EXPECT_TRUE( std::is_trivially_constructible<
                   vecmem::details::vector_data< int > >() );
   EXPECT_TRUE( std::is_trivially_copy_constructible<
                   vecmem::details::vector_data< int > >() );
}
