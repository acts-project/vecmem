/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/array.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <cstring>
#include <type_traits>
#include <vector>

/// Function testing a particular array object.
template< typename T, std::size_t N >
void test_array( vecmem::array< T, N >& a ) {

   // Make sure that we use integer types for the test, as it really only works
   // for that...
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
         assert( static_cast< std::size_t >( *itr ) == i );
      }
      auto ritr = a.rbegin();
      for( std::size_t i = a.size() - 1; ritr != a.rend(); ++ritr, --i ) {
         assert( static_cast< std::size_t >( *ritr ) == i );
      }
   }

   // Check its contents using a range based loop.
   {
      std::size_t i = 0;
      for( T value : a ) {
         assert( static_cast< std::size_t >( value ) == i++ );
      }
   }

   // Fill the array with a constant value.
   static constexpr std::size_t VALUE = 123;
   a.fill( VALUE );
   for( T value : a ) {
      assert( value == VALUE );
   }

   // Make sure that it succeeded.
   if( ! a.empty() ) {
      assert( a.front() == VALUE );
      assert( a.back() == VALUE );
   }
   const std::vector< T > reference( a.size(), VALUE );
   assert( memcmp( a.data(), reference.data(), a.size() * sizeof( T ) ) == 0 );
}

int main() {

   // The resource used throughout the test.
   vecmem::host_memory_resource resource;

   // Create an array whose size is decided at compile time.
   vecmem::array< int, 10 > a1( resource );
   test_array( a1 );

   // Create an array whose size is decided at runtime.
   vecmem::array< int > a2( resource, 20 );
   test_array( a2 );

   // Test zero sized arrays.
   vecmem::array< unsigned int, 0 > a3( resource );
   test_array( a3 );

   vecmem::array< unsigned int > a4( resource, 0 );
   test_array( a4 );

   // Return gracefully.
   return 0;
}
