/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/resources/memory_resource.hpp"

// System include(s).
#include <cstddef>

namespace vecmem::details {

   /// Struct used for deleting an allocated memory block
   ///
   /// It can be used to make things like @c std::unique_ptr talk to
   /// @c vecmem::memory_resource.
   ///
   class deleter {

   public:
      /// Constructor
      deleter( std::size_t bytes, memory_resource& resource );

      /// Copy constructor
      deleter( const deleter& ) = default;
      /// Move constructor
      deleter( deleter&& ) = default;
      /// Copy assignment operator
      deleter& operator=( const deleter& ) = default;
      /// Move assignment operator
      deleter& operator=( deleter&& ) = default;

      /// Operator performing the deletion of the object.
      void operator()( void* ptr );

   private:
      /// The number of bytes allocated for the memory block
      std::size_t m_bytes;
      /// The memory resource used for deleting the memory block
      memory_resource* m_resource;

   }; // struct deleter

} // namespace vecmem::details
