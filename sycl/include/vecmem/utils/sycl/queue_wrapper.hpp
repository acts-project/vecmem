/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <memory>
#include <string>

namespace vecmem::sycl {

   // Forward declaration(s).
   namespace details {
      class opaque_queue;
   }

   /// Wrapper class for @c cl::sycl::queue
   ///
   /// It is necessary for passing around SYCL queue objects in code that should
   /// not be directly exposed to the SYCL headers.
   ///
   class queue_wrapper {

   public:
      /// Construct a queue for a device with a specific name
      queue_wrapper( const std::string& deviceName = "" );
      /// Wrap an existing @c cl::sycl::queue object
      ///
      /// Without taking ownership of it!
      ///
      queue_wrapper( void* queue );

      /// Copy constructor
      queue_wrapper( const queue_wrapper& parent );
      /// Move constructor
      queue_wrapper( queue_wrapper&& parent );

      /// Destructor
      ~queue_wrapper();

      /// Copy assignment
      queue_wrapper& operator=( const queue_wrapper& rhs );
      /// Move assignment
      queue_wrapper& operator=( queue_wrapper&& rhs );

      /// Access a typeless pointer to the managed @c cl::sycl::queue object
      void* queue();
      /// Access a typeless pointer to the managed @c cl::sycl::queue object
      const void* queue() const;

   private:
      /// Bare pointer to the wrapped @c cl::sycl::queue object
      void* m_queue;
      /// Smart pointer to the managed @c cl::sycl::queue object
      std::unique_ptr< details::opaque_queue > m_managedQueue;

   }; // class queue_wrapper

} // namespace vecmem::sycl
