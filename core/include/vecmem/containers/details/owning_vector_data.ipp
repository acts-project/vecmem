/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem::details {

   template< typename TYPE >
   VECMEM_HOST
   owning_vector_data< TYPE >::
   owning_vector_data( std::size_t size, memory_resource& resource )
   : vector_data< TYPE >( size, nullptr ),
     m_memory( static_cast< TYPE* >( resource.allocate( size * sizeof( TYPE ) ) ),
               deleter( size * sizeof( TYPE ), resource ) ) {

      // Weirdly enough Clang doesn't understand what "m_ptr" by itself would
      // refer to... :-/
      vector_data< TYPE >::m_ptr = m_memory.get();
   }

} // namespace vecmem::details
