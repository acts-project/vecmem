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

namespace vecmem::sycl::details {

   /// SYCL memory resource base class
   ///
   /// This class is used as base by all of the oneAPI/SYCL memory resource
   /// classes. It holds functionality that those classes all need.
   ///
   class memory_resource_base : public memory_resource {

   public:
      /// Constructor on top of a user-provided queue
      memory_resource_base( const queue_wrapper& queue = { "" } );

   protected:
      /// The queue that the allocations are made for/on
      queue_wrapper m_queue;

   private:
      /// Function performing the memory de-allocation
      void do_deallocate( void* ptr, std::size_t nbytes,
                          std::size_t alignment ) override final;

      /// Function comparing two memory resource instances
      bool do_is_equal(
         const memory_resource& other ) const noexcept override final;

   }; // memory_resource_base

} // namespace vecmem::sycl::details
