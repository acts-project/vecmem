/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/host_memory_manager.hpp"

// System include(s).
#include <algorithm>
#include <cstdlib>
#include <unistd.h>

namespace vecmem {

   host_memory_manager::~host_memory_manager() {

      // Free up all pending allocations.
      for( void* ptr : m_memory ) {
         free( ptr );
      }
   }

   void host_memory_manager::set_maximum_capacity( std::size_t, int ) {}

   std::size_t host_memory_manager::available_memory( int ) const {

      // Get the available amount of memory in a POSIX way.
      const long freePages = sysconf( _SC_AVPHYS_PAGES );
      const long pageSize = sysconf( _SC_PAGE_SIZE );
      return freePages * pageSize;
   }

   void* host_memory_manager::allocate( std::size_t sizeInBytes, int ) {

      // Allocate the requested block, and note down its size.
      void* result = malloc( sizeInBytes );
      m_memory.push_back( result );
      return result;
   }

   void host_memory_manager::deallocate( void* ptr ) {

      // Remove the pointer from the internal list, and free up the memory.
      m_memory.erase( std::remove( m_memory.begin(), m_memory.end(), ptr ),
                      m_memory.end() );
      free( ptr );
      return;
   }

   void host_memory_manager::reset( int ) {

      // Free up all known allocations.
      for( void* ptr : m_memory ) {
         free( ptr );
      }
      m_memory.clear();
      return;
   }

   bool host_memory_manager::is_host_accessible() const {

      return true;
   }

} // namespace vecmem
