/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/resources/memory_resource.hpp"
#include "vecmem/utils/sycl/queue_wrapper.hpp"

namespace vecmem::sycl {

   /// Memory resource for a specific SYCL device
   ///
   /// The construction of the @c cl::sycl::queue, that the resource is meant
   /// to operate on, is left up to the user.
   ///
   class device_memory_resource : public memory_resource {

   public:
      /// Constructor on top of a user-provided queue
      device_memory_resource( const queue_wrapper& queue = { "" } );

   private:
      /// Function performing the memory allocation
      void* do_allocate( std::size_t nbytes, std::size_t alignment ) override;

      /// Function performing the memory de-allocation
      void do_deallocate( void* ptr, std::size_t nbytes,
                          std::size_t alignment ) override;

      /// Function comparing two memory resource instances
      bool do_is_equal( const memory_resource& other ) const noexcept override;

      /// The queue that the allocations are made for/on
      queue_wrapper m_queue;
   };

} // namespace vecmem::sycl
