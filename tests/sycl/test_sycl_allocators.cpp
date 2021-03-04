/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/sycl/device_memory_resource.hpp"
#include "vecmem/memory/sycl/host_memory_resource.hpp"
#include "vecmem/memory/sycl/shared_memory_resource.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>

/// Function running tests using the specified memory resource
void run_host_tests( vecmem::memory_resource& resource ) {

   // Create the test vector.
   vecmem::vector< int > testv( &resource );

   // Manipulate it in some simple ways.
   for( int i = 0; i < 100; ++i ) {
      testv.push_back( i );
   }
   assert( testv.size() == 100 );
   for( int i = 0; i < 100; ++i ) {
      assert( testv.at( i ) == i );
   }
   testv.resize( 1000 );
   for( int i = 0; i < 100; ++i ) {
      assert( testv.at( i ) == i );
   }
   return;
}

/// Function running simple tests with the specified memory resource.
void run_device_tests( vecmem::memory_resource& resource ) {

   void* p = resource.allocate( 100 );
   assert( p != nullptr );
   resource.deallocate( p, 100 );

   p = resource.allocate( 4096 );
   assert( p != nullptr );
   resource.deallocate( p, 4096 );
   return;
}

int main() {

   // Create all the available oneAPI/SYCL memory resources.
   vecmem::sycl::device_memory_resource device_resource;
   vecmem::sycl::host_memory_resource   host_resource;
   vecmem::sycl::shared_memory_resource shared_resource;

   // Run the tests with all available memory resources, that allow for access
   // to the memory from the host.
   run_host_tests( host_resource );
   run_host_tests( shared_resource );

   // Run much simpler tests with all of the resources.
   run_device_tests( device_resource );
   run_device_tests( host_resource );
   run_device_tests( shared_resource );

   // Return gracefully.
   return 0;
}
