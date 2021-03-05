/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {

   template< typename TYPE >
   template< typename ALLOC >
   VECMEM_HOST
   device_vector_data< TYPE >::
   device_vector_data( std::vector< TYPE, ALLOC >& vec )
   : m_size( vec.size() ), m_ptr( vec.data() ) {

   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   device_vector_data< TYPE >::
   device_vector_data( std::size_t size, pointer ptr )
   : m_size( size ), m_ptr( ptr ) {

   }

} // namespace vecmem
