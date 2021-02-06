/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/host_vector.hpp"
#include "vecmem/containers/static_vector.hpp"

// System include(s).
#include <algorithm>
#undef NDEBUG
#include <cassert>

int main() {

   // Create a reference host vector.
   vecmem::host_vector< int > host_vector = { 1, 2, 5, 6, 3, 6, 1, 7, 9 };

   // Test a "constant view vector" on top of it.
   const vecmem::const_device_vector< int >
      const_device_vector( host_vector.size(), host_vector.data() );
   assert( const_device_vector.size() == host_vector.size() );
   assert( std::equal( host_vector.begin(), host_vector.end(),
                       const_device_vector.begin() ) );

   // Test a "non-constant view vector" on top of it.
   vecmem::device_vector< int > device_vector( host_vector.size(),
                                               host_vector.data() );
   assert( device_vector.size() == host_vector.size() );
   assert( std::equal( host_vector.begin(), host_vector.end(),
                       device_vector.begin() ) );
   device_vector[ 2 ] = 15;
   assert( std::equal( host_vector.begin(), host_vector.end(),
                       device_vector.begin() ) );

   // Create a static sized vector with the same values as the reference.
   vecmem::static_vector< int, 20 > static_vector;
   static_vector.resize( host_vector.size() );
   std::copy( host_vector.begin(), host_vector.end(), static_vector.begin() );
   assert( static_vector.size() == host_vector.size() );
   assert( std::equal( host_vector.begin(), host_vector.end(),
                       static_vector.begin() ) );

   // Return gracefully.
   return 0;
}
