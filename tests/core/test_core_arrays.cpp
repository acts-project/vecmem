/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/array.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <cstring>
#include <type_traits>
#include <vector>

/// Test case for @c vecmem::array
///
/// It provides a reusable @c vecmem::host_memory_resource for the tests, and
/// provides a templated test function doing the heavy lifting for the test(s).
///
class core_array_test : public testing::Test {

protected:
   /// Function testing a particular array object.
   template< typename T, std::size_t N, typename A >
   void test_array( vecmem::array< T, N, A >& a ) {

      // Make sure that we use integer types for the test, as it really only
      // works for that...
      static_assert( std::is_integral< T >::value,
                     "Can only use integer types with this test" );

      // Fill the array with some simple values.
      for( std::size_t i = 0; i < a.size(); ++i ) {
         a.at( i ) = i;
      }

      // Check the contents using iterator based loops.
      {
         auto itr = a.begin();
         for( std::size_t i = 0; itr != a.end(); ++itr, ++i ) {
            EXPECT_TRUE( static_cast< std::size_t >( *itr ) == i );
         }
         auto ritr = a.rbegin();
         for( std::size_t i = a.size() - 1; ritr != a.rend(); ++ritr, --i ) {
            EXPECT_TRUE( static_cast< std::size_t >( *ritr ) == i );
         }
      }

      // Check its contents using a range based loop.
      {
         std::size_t i = 0;
         for( T value : a ) {
            EXPECT_TRUE( static_cast< std::size_t >( value ) == i++ );
         }
      }

      // Fill the array with a constant value.
      static constexpr std::size_t VALUE = 123;
      a.fill( VALUE );
      for( T value : a ) {
         EXPECT_TRUE( value == VALUE );
      }

      // Make sure that it succeeded.
      if( ! a.empty() ) {
         EXPECT_TRUE( a.front() == VALUE );
         EXPECT_TRUE( a.back() == VALUE );
      }
      const std::vector< T > reference( a.size(), VALUE );
      EXPECT_TRUE( memcmp( a.data(), reference.data(), a.size() * sizeof( T ) )
                   == 0 );
   }

   /// The resource used throughout the test.
   vecmem::host_memory_resource m_resource;

}; // class core_array_test

/// Test with a non-zero sized array whose size is fixed at compile time.
TEST_F( core_array_test, non_zero_compile_time ) {

   vecmem::array< int, 10 > a( &m_resource );
   test_array( a );
}

/// Test with a non-zero sized array whose size is specified at runtime
TEST_F( core_array_test, non_zero_runtime ) {

   vecmem::array< int > a( &m_resource, 20 );
   test_array( a );
}

/// Test with a zero sized array whose size is fixed at compile time.
TEST_F( core_array_test, zero_compile_time ) {

   vecmem::array< unsigned int, 0 > a( &m_resource );
   test_array( a );
}

/// Test with a zero sized array whose size is specified at runtime
TEST_F( core_array_test, zero_runtime ) {

   vecmem::array< unsigned int > a( &m_resource, 0 );
   test_array( a );
}
