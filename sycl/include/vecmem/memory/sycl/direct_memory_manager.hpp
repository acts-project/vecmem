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

// Forward declaration(s).
inline namespace cl { namespace sycl {
   class queue;
} }

namespace vecmem::sycl {

   /// A very simple memory manager using SYCL USM memory functions directly
   class direct_memory_manager : public memory_manager_interface {

   public:
      /// Memory type managed by the object
      enum class memory_type {
         device = 1, ///< Memory allocated in device memory
         host   = 2, ///< Pinned memory allocated on the host
         shared = 3  ///< SYCL shared memory allocated on the device and host
      };

      /// Constructor, allocating the default amount of memory
      direct_memory_manager( memory_type type = memory_type::shared,
                             std::size_t sizeInBytes = 200 * 1024l * 1024l );
      /// Destructor, freeing up all allocations
      ~direct_memory_manager();

      /// Default device, configurable by the user
      static constexpr int DEFAULT_DEVICE = -1;

      /// @name Custom function(s)
      /// @{

      /// Associate SYCL queues with integer "device IDs"
      void set_queue( int device, const cl::sycl::queue& queue );

      /// @}

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
         /// Explicit constructor
         device_memory();
         /// Explicit copy constructor
         device_memory( const device_memory& parent );
         /// Explicit move constructor
         device_memory( device_memory&& parent );
         /// Explicit destructor
         ~device_memory();
         /// Explicit copy assignment
         device_memory& operator=( const device_memory& rhs );
         /// Explicit move assignment
         device_memory& operator=( device_memory&& rhs );

         /// Queue object for which the allocation is made
         std::unique_ptr< cl::sycl::queue > m_queue;
         /// List of memory allocations on the device
         std::list< void* > m_ptrs;
      };

      /// Access the object describing the memory allocation on a specific
      /// device (const)
      const device_memory& get_device_memory( int device ) const;
      /// Access the object describing the memory allocation on a specific
      /// device (non-const)
      device_memory& get_device_memory( int device );

      /// The memory type managed by the object
      memory_type m_type;
      /// Object holding information about memory allocations on all devices
      std::vector< device_memory > m_memory;
      /// Object holding information about memory allocations on the default
      /// device
      device_memory m_defaultMemory;

   }; // class direct_memory_manager

} // namespace vecmem::sycl
