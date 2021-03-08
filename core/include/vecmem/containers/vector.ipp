/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {

   template< typename TYPE, typename ALLOC >
   VECMEM_HOST
   device_vector_data< TYPE >
   get_data( std::vector< TYPE, ALLOC >& vec ) {

      return device_vector_data< TYPE >( vec.size(), vec.data() );
   }

   template< typename TYPE, typename ALLOC >
   VECMEM_HOST
   device_vector_data< const TYPE >
   get_data( const std::vector< TYPE, ALLOC >& vec ) {

      return device_vector_data< const TYPE >( vec.size(), vec.data() );
   }

} // namespace vecmem
