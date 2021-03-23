/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/deallocator.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <memory>
#include <type_traits>

namespace vecmem::details {

   /// Object owning the data held by it
   ///
   /// This can come in handy in a number of cases, especially when using
   /// device-only memory blocks.
   ///
   template< typename TYPE >
   class vector_buffer : public vector_view< TYPE > {

   public:
      /// The base type used by this class
      typedef vector_view< TYPE > base_type;

      /// @name Checks on the type of the array element
      /// @{

      /// Make sure that the template type does not have a custom destructor
      static_assert( std::is_trivially_destructible< TYPE >::value,
                     "vecmem::details::vector_buffer can not handle types with "
                     "custom destructors" );

      /// @}

      /// Constructor with a size
      ///
      /// It is left up to external code to copy data into the allocated memory.
      ///
      VECMEM_HOST
      vector_buffer( std::size_t size, memory_resource& resource );

   private:
      /// Data object owning the allocated memory
      std::unique_ptr< TYPE, deallocator > m_memory;

   }; // class vector_buffer

} // namespace vecmem::details

// Include the implementation.
#include "vecmem/containers/impl/vector_buffer.ipp"
