/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/allocators/allocator.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/memory_manager.hpp"
#include "vecmem/memory/cuda/arena_memory_manager.hpp"
#include "vecmem/memory/cuda/direct_memory_manager.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>

/// Function running tests with the active memory manager.
void run_host_tests(vecmem::memory_resource & resource) {

   // Create the test vector.
   vecmem::vector<int> testv(&resource);

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

int main() {
   vecmem::cuda::host_memory_resource host_resource;
   vecmem::cuda::managed_memory_resource managed_resource;

   // Run the tests with all available memory managers, that allow for access to
   // the memory from the host.
   run_host_tests(host_resource);
   run_host_tests(managed_resource);

   // Return gracefully.
   return 0;
}
