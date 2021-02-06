/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <utility>

namespace vecmem {

   /// Abstract interface for any type of memory manager
   class memory_manager_interface {

   public:
      /// Virtual destructor, to make the vtable happy
      virtual ~memory_manager_interface() = default;

      /// Set the amount of memory to use on a particular device
      virtual void set_maximum_capacity( std::size_t sizeInBytes,
                                         int device = -1 ) = 0;

      /// Get the amount of memory still available on a specific device
      virtual std::size_t available_memory( int device = -1 ) const = 0;

      /// Get a pointer to an available memory block on a specific device
      virtual void* allocate( std::size_t sizeInBytes, int device = -1 ) = 0;

      /// Deallocate a specific memory block
      virtual void deallocate( void* ptr ) = 0;

      /// Reset all allocations on a given device
      virtual void reset( int device = -1 ) = 0;

   }; // class memory_manager_interface

} // namespace vecmem
