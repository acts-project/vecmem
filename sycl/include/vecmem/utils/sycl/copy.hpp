/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/sycl/queue_wrapper.hpp"

namespace vecmem::sycl {

   /// Specialisation of @c vecmem::copy for SYCL
   ///
   /// Unlike @c vecmem::cuda::copy and @c vecmem::hip::copy, this object does
   /// have a state. As USM memory operations in SYCL happen through a
   /// @c cl::sycl::queue object. So this object needs to point to a valid
   /// queue object itself.
   ///
   class copy : public vecmem::copy {

   public:
      /// Constructor on top of a user-provided queue
      copy( const queue_wrapper& queue = { "" } );

   protected:
      /// Perform a memory copy using SYCL
      virtual void do_copy( std::size_t size, const void* from, void* to,
                            type::copy_type cptype ) override;
      /// Fill a memory area using SYCL
      virtual void do_memset( std::size_t size, void* ptr, int value ) override;

   private:
      /// The queue that the copy operations are made with/for
      queue_wrapper m_queue;

   }; // class copy

} // namespace vecmem::sycl
