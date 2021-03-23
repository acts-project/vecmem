/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace {

   /// Function creating the smart pointer for @c vecmem::data::vector_buffer
   template< typename TYPE >
   std::unique_ptr< TYPE, vecmem::details::deallocator >
   allocate_buffer_memory(
      typename vecmem::data::vector_buffer< TYPE >::size_type size,
      vecmem::memory_resource& resource ) {

      const typename vecmem::data::vector_buffer< TYPE >::size_type
         byteSize = size * sizeof( TYPE );
      return { size == 0 ? nullptr :
               static_cast< TYPE* >( resource.allocate( byteSize ) ),
               { byteSize, resource } };
   }

} // private namespace

namespace vecmem::data {

   template< typename TYPE >
   VECMEM_HOST
   vector_buffer< TYPE >::
   vector_buffer( std::size_t size, memory_resource& resource )
   : base_type( size, nullptr ),
     m_memory( ::allocate_buffer_memory< TYPE >( size, resource ) ) {

      // Weirdly enough Clang doesn't understand what "m_ptr" by itself would
      // refer to... :-/
      base_type::m_ptr = m_memory.get();
   }

} // namespace vecmem::data
