/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/static_vector.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <cmath>
#include <cstddef>

/// Test case for @c vecmem::static_vector
template< typename T >
class core_static_vector_test : public testing::Test {};

/// Test suite for primitive types.
typedef testing::Types< int, long, float, double > primitive_types;
TYPED_TEST_SUITE( core_static_vector_test, primitive_types );

/// Test the constructor with a size
TYPED_TEST( core_static_vector_test, sized_constructor ) {

   // Create a vector.
   vecmem::static_vector< TypeParam, 100 > v( 10 );
   EXPECT_EQ( v.size(), 10 );

   // Make sure that it's elements were created as expected.
   for( const TypeParam& value : v ) {
      EXPECT_EQ( value, TypeParam() );
   }
}

/// Test the constructor with a size and a custom value
TYPED_TEST( core_static_vector_test, sized_constructor_with_value ) {

   // Create a vector.
   static const TypeParam DEFAULT_VALUE = 10;
   vecmem::static_vector< TypeParam, 100 > v( 10, DEFAULT_VALUE );
   EXPECT_EQ( v.size(), 10 );

   // Make sure that it's elements were created as expected.
   for( const TypeParam& value : v ) {
      EXPECT_EQ( value, DEFAULT_VALUE );
   }
}

/// Test the copy constructor
TYPED_TEST( core_static_vector_test, copy_constructor ) {

   /// Create a reference vector.
   static const TypeParam DEFAULT_VALUE = 123;
   vecmem::static_vector< TypeParam, 100 > ref( 10, DEFAULT_VALUE );

   // Create a copy.
   vecmem::static_vector< TypeParam, 100 > copy( ref );

   // Check the copy.
   EXPECT_EQ( ref.size(), copy.size() );
   EXPECT_TRUE( std::equal( ref.begin(), ref.end(), copy.begin(),
                            []( const TypeParam& v1, const TypeParam& v2 ) {
                               return ( std::abs( v1 - v2 ) < 0.001 );
                            } ) );
}
