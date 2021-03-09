/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem::details {

   template< typename T, typename Allocator >
   deleter< T, Allocator >::
   deleter( size_type elements, const allocator_type& allocator )
   : m_elements( elements ), m_allocator( allocator ) {

   }

   template< typename T, typename Allocator >
   void deleter< T, Allocator >::operator()( void* ptr ) {

      m_allocator.deallocate( static_cast< T* >( ptr ), m_elements );
   }

} // namespace vecmem::details
