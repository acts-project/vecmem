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
#include <memory>
#include <vector>

namespace vecmem { namespace hip {

   /// A very simple memory manager using HIP memory functions directly
   class direct_memory_manager : public memory_manager_interface {

   public:
      /// Memory type managed by the object
      enum class memory_type {
         device     = 1, ///< Memory allocated in device memory
         host       = 2  ///< Pinned memory allocated on the host, can be made
                         ///  available on the device
      };

      /// Constructor, allocating the default amount of memory
      direct_memory_manager( memory_type type = memory_type::host,
                             std::size_t sizeInBytes = 200 * 1024l * 1024l );
      /// Destructor, freeing up all allocations
      ~direct_memory_manager();

      /// Default device, configurable by the user
      static constexpr int DEFAULT_DEVICE = -1;

      /// @name Functions inherited from @c detraydm::memory_manager_interface
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
      /// Struct describing the state of the memory allocation on a particular
      /// device
      struct device_memory {
         /// List of memory allocations on the device
         std::list< void* > m_ptrs;
      };

      /// Create a valid device ID from what the user provided
      static void get_device( int& device );
      /// Access the object describing the memory allocation on a specific
      /// device
      device_memory& get_device_memory( int& device );

      /// The memory type managed by the object
      memory_type m_type;
      /// Object holding information about memory allocations on all devices
      std::vector< device_memory > m_memory;

   }; // class direct_memory_manager

} } // namespace vecmem::hip
