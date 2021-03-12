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
#include <vector>

/// Test case for @c vecmem::static_vector
template< typename T >
class core_static_vector_test : public testing::Test {};

/// Test suite for primitive types.
typedef testing::Types< int, long, float, double > primitive_types;
TYPED_TEST_SUITE( core_static_vector_test, primitive_types );

namespace {
   /// Helper function for comparing the value of primitive types
   template< typename T >
   bool almost_equal( const T& v1, const T& v2 ) {
      return ( std::abs( v1 - v2 ) < 0.001 );
   }
} // private namespace

/// Helper macro for comparing two vectors.
#define EXPECT_EQ_VEC( v1, v2 )                                                \
   EXPECT_EQ( v1.size(), v2.size() );                                          \
   EXPECT_TRUE( std::equal( std::begin( v1 ), std::end( v1 ), std::begin( v2 ),\
                            ::almost_equal< TypeParam > ) )

/// Test the constructor with a size
TYPED_TEST( core_static_vector_test, constructor_with_size ) {

   // Create a vector.
   vecmem::static_vector< TypeParam, 100 > v( 10 );
   EXPECT_EQ( v.size(), 10 );

   // Make sure that it's elements were created as expected.
   for( const TypeParam& value : v ) {
      EXPECT_EQ( value, TypeParam() );
   }
}

/// Test the constructor with a size and a custom value
TYPED_TEST( core_static_vector_test, constructor_with_size_and_value ) {

   // Create a vector.
   static const TypeParam DEFAULT_VALUE = 10;
   vecmem::static_vector< TypeParam, 100 > v( 10, DEFAULT_VALUE );
   EXPECT_EQ( v.size(), 10 );

   // Make sure that it's elements were created as expected.
   for( const TypeParam& value : v ) {
      EXPECT_EQ( value, DEFAULT_VALUE );
   }
}

/// Test the constructor with a range of values
TYPED_TEST( core_static_vector_test, constructor_with_iterators ) {

   // Create a reference vector.
   const std::vector< TypeParam > ref = { 1, 23, 64, 66, 23, 64, 99 };

   // Create the test vector based on it.
   const vecmem::static_vector< TypeParam, 100 > test( ref.begin(), ref.end() );
   EXPECT_EQ_VEC( ref, test );
}

/// Test the default constructor
TYPED_TEST( core_static_vector_test, default_constructor ) {

   // Create a default vector.
   const vecmem::static_vector< TypeParam, 100 > test1;
   EXPECT_EQ( test1.size(), 0 );

   // Create a default vector with zero capacity.
   const vecmem::static_vector< TypeParam, 0 > test2;
   EXPECT_EQ( test2.size(), 0 );
}

/// Test the copy constructor
TYPED_TEST( core_static_vector_test, copy_constructor ) {

   // Create a reference vector.
   static const TypeParam DEFAULT_VALUE = 123;
   vecmem::static_vector< TypeParam, 100 > ref( 10, DEFAULT_VALUE );

   // Create a copy.
   vecmem::static_vector< TypeParam, 100 > copy( ref );

   // Check the copy.
   EXPECT_EQ_VEC( ref, copy );
}

/// Test the element access functions and operators
TYPED_TEST( core_static_vector_test, element_access ) {

   // Create a vector.
   vecmem::static_vector< TypeParam, 100 > v( 10 );

   // Modify its elements.
   for( std::size_t i = 0; i < v.size(); ++i ) {
      v.at( i ) = TypeParam( i );
   }

   // Check that the settings "took".
   for( std::size_t i = 0; i < v.size(); ++i ) {
      EXPECT_EQ( v[ i ], TypeParam( i ) );
   }

   // Test the front() and back() functions.
   EXPECT_EQ( v.front(), TypeParam( 0 ) );
   EXPECT_EQ( v.back(), TypeParam( 9 ) );

   // Make sure that the vector points to a meaningful place.
   EXPECT_EQ( &( v.front() ), v.data() );
}

/// Test modifying an existing vector
TYPED_TEST( core_static_vector_test, modifications ) {

   // Here we perform the same operations on a reference and on a test vector.
   // Assuming that std::vector would always behave correctly, so we just need
   // to test vecmem::static_vector against it.

   // Fill the vectors with some simple content.
   std::vector< TypeParam > ref( 50 );
   vecmem::static_vector< TypeParam, 100 > test( 50 );
   EXPECT_EQ_VEC( ref, test );
   for( std::size_t i = 0; i < ref.size(); ++i ) {
      ref[ i ] = TypeParam( i );
      test[ i ] = TypeParam( i );
   }
   EXPECT_EQ_VEC( ref, test );

   // Add a single element to the end of them.
   ref.push_back( TypeParam( 60 ) );
   test.push_back( TypeParam( 60 ) );
   EXPECT_EQ_VEC( ref, test );

   // Do the same, just in a slightly different colour.
   ref.emplace_back( 70 );
   test.emplace_back( 70 );
   EXPECT_EQ_VEC( ref, test );

   // Insert a single element in the middle of them.
   ref.insert( ref.begin() + 20, TypeParam( 15 ) );
   test.insert( test.begin() + 20, TypeParam( 15 ) );
   EXPECT_EQ_VEC( ref, test );

   // Emplace a single element in the middle of them.
   ref.emplace( ref.begin() + 15, 55 );
   test.emplace( test.begin() + 15, 55 );
   EXPECT_EQ_VEC( ref, test );

   // Remove one element from them.
   ref.erase( ref.begin() + 30 );
   test.erase( test.begin() + 30 );
   EXPECT_EQ_VEC( ref, test );

   // Remove a range of elements from them.
   ref.erase( ref.begin() + 10, ref.begin() + 25 );
   test.erase( test.begin() + 10, test.begin() + 25 );
   EXPECT_EQ_VEC( ref, test );

   // Insert N copies of the same value in them.
   ref.insert( ref.begin() + 13, 10, TypeParam( 34 ) );
   test.insert( test.begin() + 13, 10, TypeParam( 34 ) );
   EXPECT_EQ_VEC( ref, test );

   // Insert a range of new values.
   static const std::vector< TypeParam > ins = { 33, 44, 55 };
   ref.insert( ref.begin() + 24, ins.begin(), ins.end() );
   test.insert( test.begin() + 24, ins.begin(), ins.end() );
   EXPECT_EQ_VEC( ref, test );

   // Reduce the size of them.
   ref.resize( 5 );
   test.resize( 5 );
   EXPECT_EQ_VEC( ref, test );

   // Expand them.
   ref.resize( 20 );
   test.resize( 20 );
   EXPECT_EQ_VEC( ref, test );

   // Expand them using a specific fill value.
   ref.resize( 30, TypeParam( 93 ) );
   test.resize( 30, TypeParam( 93 ) );
   EXPECT_EQ_VEC( ref, test );
}

/// Test the capacity functions of @c vecmem::static_vector
TYPED_TEST( core_static_vector_test, capacity ) {

   // Create a simple vector.
   vecmem::static_vector< TypeParam, 100 > v;

   // Simple checks on the empty vector.
   EXPECT_EQ( v.empty(), true );
   EXPECT_EQ( v.size(), 0 );
   EXPECT_EQ( v.max_size(), 100 );
   EXPECT_EQ( v.capacity(), 100 );

   // Resize it, and test it again.
   v.resize( 50 );
   EXPECT_EQ( v.empty(), false );
   EXPECT_EQ( v.size(), 50 );
   EXPECT_EQ( v.max_size(), 100 );
   EXPECT_EQ( v.capacity(), 100 );

   // Make sure that the (no-op) reserve function can be called.
   v.reserve( 70 );
}
