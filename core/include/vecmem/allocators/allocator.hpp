/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_manager.hpp"
#include "vecmem/memory/memory_manager_interface.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {

   /// STL allocator using the "currently active" memory manager
   template< typename TYPE >
   class allocator {

   public:
      /// @name Type definitions that need to be provided by the allocator
      /// @{
      typedef std::size_t    size_type;
      typedef std::ptrdiff_t difference_type;
      typedef TYPE*          pointer;
      typedef const TYPE*    const_pointer;
      typedef TYPE&          reference;
      typedef const TYPE&    const_reference;
      typedef TYPE           value_type;
      /// @}

      /// @name "Behaviour declarations" for the allocator
      /// @{
      typedef std::true_type propagate_on_container_move_assignment;
      typedef std::true_type is_always_equal;
      /// @}

      /// Allocate a requested amount of memory
      pointer allocate( size_type n, const void* = nullptr ) {
         memory_manager_interface& mmgr = memory_manager::instance().get();
         return static_cast< pointer >(
            mmgr.allocate( n * sizeof( value_type ) ) );
      }

      /// Deallocate a previously allocated block of memory
      void deallocate( pointer ptr, size_type ) {
         memory_manager_interface& mmgr = memory_manager::instance().get();
         mmgr.deallocate( ptr );
      }

      /// Only initialise the memory if it is host-accessible.
      template< typename U, typename... Args >
      void construct( U* ptr, Args&&... args ) {
         memory_manager_interface& mmgr = memory_manager::instance().get();
         if( mmgr.is_host_accessible() ) {
            new( ptr ) U( std::forward< Args >( args )... );
         }
         return;
      }

      /// Only destroy objects in host-accessible memory.
      template< typename U >
      void destroy( U* ptr ) {
         memory_manager_interface& mmgr = memory_manager::instance().get();
         if( mmgr.is_host_accessible() ) {
            ptr->~U();
         }
      }

   }; // class allocator

} // namespace vecmem
