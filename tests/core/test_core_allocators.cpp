/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/allocators/allocator.hpp"

// System include(s).
#include <algorithm>
#undef NDEBUG
#include <cassert>
#include <vector>

// Vector type using the generic vecmem allocator.
template< typename T >
using custom_vector = std::vector< T, vecmem::allocator< T > >;

int main() {

   // Create a "custom vector" and a reference vector, and do some tests with
   // them.
   std::vector< int > reference_vector;
   reference_vector.reserve( 100 );
   custom_vector< int > test_vector;
   test_vector.reserve( 100 );

   for( int i = 0; i < 20; ++i ) {
      reference_vector.push_back( i * 2 );
      test_vector.push_back( i * 2 );
   }
   assert( reference_vector.size() == test_vector.size() );
   assert( std::equal( reference_vector.begin(), reference_vector.end(),
                       test_vector.begin() ) );

   for( int i : { 26, 38, 25 } ) {
      std::remove( reference_vector.begin(), reference_vector.end(), i );
      std::remove( test_vector.begin(), test_vector.end(), i );
   }
   assert( reference_vector.size() == test_vector.size() );
   assert( std::equal( reference_vector.begin(), reference_vector.end(),
                       test_vector.begin() ) );

   // Return gracefully.
   return 0;
}
