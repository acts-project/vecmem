/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/device_vector.hpp"
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
                   vecmem::data::jagged_vector_view< const int > >() );
   EXPECT_TRUE( std::is_trivially_constructible<
                   vecmem::data::jagged_vector_view< const int > >() );
   EXPECT_TRUE( std::is_trivially_copy_constructible<
                   vecmem::data::jagged_vector_view< const int > >() );

   EXPECT_TRUE( std::is_trivially_default_constructible<
                   vecmem::data::jagged_vector_view< int > >() );
   EXPECT_TRUE( std::is_trivially_constructible<
                   vecmem::data::jagged_vector_view< int > >() );
   EXPECT_TRUE( std::is_trivially_copy_constructible<
                   vecmem::data::jagged_vector_view< int > >() );

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

namespace {

   /// Type used in the tests
   struct TestType {
      double a, b;
      int c, d;
   };

   /// Type providing access to @c vecmem::device_vector's internals
   template< typename T >
   class test_device_vector : public vecmem::device_vector< T > {
   public:
      /// Type of the base class
      typedef vecmem::device_vector< T > base_type;
      /// Inherit the base's constructor(s).
      using base_type::base_type;
      /// Position of @c m_size wrt. the object
      std::ptrdiff_t size_pos() const {
         return ( reinterpret_cast< const char* >( &( base_type::m_size ) ) -
                  reinterpret_cast< const char* >( this ) );
      }
      /// Position of @c m_ptr wrt. the object
      std::ptrdiff_t ptr_pos() const {
         return ( reinterpret_cast< const char* >( &( base_type::m_ptr ) ) -
                  reinterpret_cast< const char* >( this ) );
      }
   };

} // private namespace

/// Test the "compatibility" of @c vecmem::device_vector and @c vecmem::data::vector_view
///
/// The @c vecmem::jagged_device_vector code makes use of the fact that arrays
/// of these two can be converted into each other. This test tries to make sure
/// that this would remain the case.
///
TEST_F( core_device_container_test, type_equivalence ) {

   {
      vecmem::data::vector_view< int > data;
      test_device_vector< int > vector( data );
      EXPECT_EQ( sizeof( data ), sizeof( vector ) );
      EXPECT_EQ( ( reinterpret_cast< char* >( &( data.m_size ) ) -
                   reinterpret_cast< char* >( &data ) ),
                 vector.size_pos() );
      EXPECT_EQ( ( reinterpret_cast< char* >( &( data.m_ptr ) ) -
                   reinterpret_cast< char* >( &data ) ),
                 vector.ptr_pos() );
   }
   {
      vecmem::data::vector_view< TestType > data;
      test_device_vector< TestType > vector( data );
      EXPECT_EQ( sizeof( data ), sizeof( vector ) );
      EXPECT_EQ( ( reinterpret_cast< char* >( &( data.m_size ) ) -
                   reinterpret_cast< char* >( &data ) ),
                 vector.size_pos() );
      EXPECT_EQ( ( reinterpret_cast< char* >( &( data.m_ptr ) ) -
                   reinterpret_cast< char* >( &data ) ),
                 vector.ptr_pos() );
   }
}
