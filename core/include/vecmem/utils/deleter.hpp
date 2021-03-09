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

   /// Struct used for deleting an allocated, host-accessible memory block
   ///
   /// It can be used to make things like @c std::unique_ptr talk to
   /// @c vecmem::memory_resource.
   ///
   template< typename T,
             typename Allocator = polymorphic_allocator< T > >
   class deleter {

   public:
      /// Size type for the number of elements to delete
      typedef std::size_t size_type;
      /// The allocator type to use
      typedef Allocator   allocator_type;

      /// Constructor
      deleter( size_type elements, const allocator_type& allocator );

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
      /// The number of elements allocated for the memory block
      size_type m_elements;
      /// The allocator used to delete the elements
      allocator_type m_allocator;

   }; // class deleter

} // namespace vecmem::details

// Include the implementation.
#include "vecmem/utils/deleter.ipp"
