/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/array.hpp"
#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/static_vector.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <numeric>

/// Test case for the custom container types
class core_container_test : public testing::Test {

protected:
   /// Memory resource to use in the tests
   vecmem::host_memory_resource m_resource;
   /// Test vector used for testing all of the custom containers
   vecmem::vector< int > m_reference_vector = { 1, 2, 5, 6, 3, 6, 1, 7, 9 };

}; // class core_container_test

/// Test(s) for @c vecmem::const_device_vector
TEST_F( core_container_test, const_device_vector ) {

   const vecmem::const_device_vector< int >
      test_vector( vecmem::get_data( m_reference_vector ) );
   EXPECT_TRUE( test_vector.size() == m_reference_vector.size() );
   EXPECT_TRUE( test_vector.empty() == m_reference_vector.empty() );
   EXPECT_TRUE( std::equal( m_reference_vector.begin(),
                            m_reference_vector.end(),
                            test_vector.begin() ) );
   EXPECT_TRUE( std::accumulate( test_vector.begin(), test_vector.end(), 0 ) ==
                std::accumulate( test_vector.rbegin(), test_vector.rend(), 0 ) );
   for( std::size_t i = 0; i < m_reference_vector.size(); ++i ) {
      EXPECT_TRUE( test_vector.at( i ) == m_reference_vector.at( i ) );
      EXPECT_TRUE( test_vector[ i ] == m_reference_vector[ i ] );
   }
}

/// Test(s) for @c vecmem::device_vector
TEST_F( core_container_test, device_vector ) {

   const vecmem::device_vector< int >
      test_vector( vecmem::get_data( m_reference_vector ) );
   EXPECT_TRUE( test_vector.size() == m_reference_vector.size() );
   EXPECT_TRUE( test_vector.empty() == m_reference_vector.empty() );
   EXPECT_TRUE( std::equal( m_reference_vector.begin(),
                            m_reference_vector.end(),
                            test_vector.begin() ) );
   EXPECT_TRUE( std::accumulate( test_vector.begin(), test_vector.end(), 0 ) ==
                std::accumulate( test_vector.rbegin(), test_vector.rend(), 0 ) );
   for( std::size_t i = 0; i < m_reference_vector.size(); ++i ) {
      EXPECT_TRUE( test_vector.at( i ) == m_reference_vector.at( i ) );
      EXPECT_TRUE( test_vector[ i ] == m_reference_vector[ i ] );
   }
}

/// Test(s) for @c vecmem::static_vector
TEST_F( core_container_test, static_vector ) {

   vecmem::static_vector< int, 20 > test_vector;
   test_vector.resize( m_reference_vector.size() );
   std::copy( m_reference_vector.begin(), m_reference_vector.end(),
              test_vector.begin() );
   EXPECT_TRUE( test_vector.size() == m_reference_vector.size() );
   EXPECT_TRUE( std::equal( m_reference_vector.begin(),
                            m_reference_vector.end(),
                            test_vector.begin() ) );
}

/// Test(s) for @c vecmem::array
TEST_F( core_container_test, array ) {

   vecmem::array< int, 20 > test_array( m_resource );
   std::copy( m_reference_vector.begin(), m_reference_vector.end(),
              test_array.begin() );
   EXPECT_TRUE( std::equal( m_reference_vector.begin(),
                            m_reference_vector.end(),
                            test_array.begin() ) );
}
