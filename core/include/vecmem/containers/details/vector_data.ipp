/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem { namespace details {

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   vector_data< TYPE >::vector_data( std::size_t size, pointer ptr )
   : m_size( size ), m_ptr( ptr ) {

   }

   template< typename TYPE >
   template< typename OTHERTYPE,
             std::enable_if_t<
                ( ! std::is_same< TYPE, OTHERTYPE >::value ) &&
                std::is_same< TYPE,
                              typename std::add_const< OTHERTYPE >::type >::value,
                bool > >
   VECMEM_HOST_AND_DEVICE
   vector_data< TYPE >::vector_data( const vector_data< OTHERTYPE >& parent )
   : m_size( parent.m_size ), m_ptr( parent.m_ptr ) {

   }

} } // namespace vecmem::details
