/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/deleter.hpp"

namespace vecmem::details {

   deleter::deleter( std::size_t bytes, memory_resource& resource )
   : m_bytes( bytes ), m_resource( &resource ) {

   }

   void deleter::operator()( void* ptr ) {

      if( ptr != nullptr ) {
         m_resource->deallocate( ptr, m_bytes );
      }
   }

} // namespace vecmem::details
