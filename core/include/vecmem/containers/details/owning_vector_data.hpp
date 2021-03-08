/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/details/vector_data.hpp"
#include "vecmem/memory/resources/memory_resource.hpp"
#include "vecmem/utils/deleter.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <memory>

namespace vecmem::details {

   /// Object owning the data held by it
   ///
   /// This can come in handy in a number of cases, especially when using
   /// device-only memory blocks.
   ///
   template< typename TYPE >
   class owning_vector_data : public vector_data< TYPE > {

   public:
      /// Constructor with a size
      ///
      /// It is left up to external code to copy data into the allocated memory.
      ///
      VECMEM_HOST
      owning_vector_data( std::size_t size, memory_resource& resource );

   private:
      /// Data object owning the allocated memory
      std::unique_ptr< TYPE, deleter > m_memory;

   }; // class owning_vector_data

} // namespace vecmem::details

// Include the implementation.
#include "vecmem/containers/details/owning_vector_data.ipp"
