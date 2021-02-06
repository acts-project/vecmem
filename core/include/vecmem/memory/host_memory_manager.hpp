/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_manager_interface.hpp"

// System include(s).
#include <list>

namespace vecmem {

   /// Simple memory manager for the host
   class host_memory_manager : public memory_manager_interface {

   public:
      /// Destructor, freeing up all allocations
      ~host_memory_manager();

      /// @name Functions inherited from @c vecmem::memory_manager_interface
      /// @{

      /// Set the amount of memory to use on a particular device
      void set_maximum_capacity( std::size_t sizeInBytes, int device ) override;

      /// Get the amount of memory still available on a specific device
      std::size_t available_memory( int device ) const override;

      /// Get a pointer to an available memory block on a specific device
      void* allocate( std::size_t sizeInBytes, int device ) override;

      /// Deallocate a specific memory block
      void deallocate( void* ptr ) override;

      /// Reset all allocations on a given device
      void reset( int device ) override;

      /// Check whether the memory allocated is accessible from the host
      bool is_host_accessible() const override;

      /// @}

   private:
      /// All memory allocations managed by this object on the host
      std::list< void* > m_memory;

   }; // class host_memory_manager

} // namespace vecmem
