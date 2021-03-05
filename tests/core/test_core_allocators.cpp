/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/contiguous_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <type_traits>
#include <vector>

/// Basic test case for the core memory resources
///
/// This just makes sure that the memory resources defined in the
/// @c vecmem::core library are more-or-less functional. Detailed tests of the
/// different memory resources are implemented in other test cases.
///
class core_memory_resource_test : public testing::Test {

protected:
   /// Function performing some basic tests using @c vecmem::vector
   template< typename T >
   void test_resource( vecmem::vector< T >& test_vector ) {

      // Make sure that we use integer types for the test, as it really only
      // works for that...
      static_assert( std::is_integral< T >::value,
                     "Can only use integer types with this test" );

      // Set up the test vector, and create a reference vector.
      std::vector< T > reference_vector;
      reference_vector.reserve( 100 );
      test_vector.reserve( 100 );

      // Fill them up with some dummy content.
      for( int i = 0; i < 20; ++i ) {
         reference_vector.push_back( i * 2 );
         test_vector.push_back( i * 2 );
      }
      // Make sure that they are the same.
      EXPECT_TRUE( reference_vector.size() == test_vector.size() );
      EXPECT_TRUE( std::equal( reference_vector.begin(), reference_vector.end(),
                               test_vector.begin() ) );

      // Remove a couple of elements from the vectors.
      for( int i : { 26, 38, 25 } ) {
         std::remove( reference_vector.begin(), reference_vector.end(), i );
         std::remove( test_vector.begin(), test_vector.end(), i );
      }
      // Make sure that they are still the same.
      EXPECT_TRUE( reference_vector.size() == test_vector.size() );
      EXPECT_TRUE( std::equal( reference_vector.begin(), reference_vector.end(),
                               test_vector.begin() ) );
   }

}; // class core_memory_resource_test

/// Test with the basic host memory resource
TEST_F( core_memory_resource_test, host ) {

   vecmem::host_memory_resource resource;
   vecmem::vector< int > test_vector( &resource );
   test_resource( test_vector );
}

/// Test with the binary page memory resource
TEST_F( core_memory_resource_test, binary_page ) {

   vecmem::host_memory_resource resource1;
   vecmem::binary_page_memory_resource resource2( resource1 );
   vecmem::vector< unsigned int > test_vector( &resource2 );
   test_resource( test_vector );
}

/// Test with the contiguous memory resource
TEST_F( core_memory_resource_test, contiguous ) {

   vecmem::host_memory_resource resource1;
   vecmem::contiguous_memory_resource resource2( resource1, 16384 );
   vecmem::vector< long > test_vector( &resource2 );
   test_resource( test_vector );
}
