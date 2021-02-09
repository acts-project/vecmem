/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/sycl/direct_memory_manager.hpp"
#include "vecmem/utils/sycl/device_selector.hpp"

// SYCL include(s).
#include <CL/sycl.hpp>

// System include(s).
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace vecmem::sycl {

   direct_memory_manager::direct_memory_manager( memory_type type,
                                                 std::size_t sizeInBytes )
   : m_type( type ) {

      // Create a default queue for the default device.
      m_defaultMemory.m_queue =
         std::make_unique< cl::sycl::queue >( device_selector() );

      // Allocate the requested amount of memory on the default device.
      set_maximum_capacity( sizeInBytes, DEFAULT_DEVICE );
   }

   direct_memory_manager::~direct_memory_manager() {}

   void direct_memory_manager::set_queue( int device,
                                          const cl::sycl::queue& queue ) {

      // Decide which device object to use. Do not use the
      // get_device_memory(...) function, as that requires a queue to already
      // be set on the object.
      device_memory* mem = nullptr;
      if( device == DEFAULT_DEVICE ) {
         mem = &m_defaultMemory;
      } else {
         if( static_cast< std::size_t >( device ) >= m_memory.size() ) {
            m_memory.resize( device + 1 );
         }
         mem = &( m_memory[ device ] );
      }

      // Check if it already has memory allocated.
      if( mem->m_ptrs.size() != 0 ) {
         const std::string deviceName =
            ( device == DEFAULT_DEVICE ? "the default device" :
              "device " + std::to_string( device ) );
         throw std::runtime_error( "Modifying " + deviceName + " after it "
                                   "already had memory allocated for it" );
      }

      // Set up a queue for the device.
      mem->m_queue = std::make_unique< cl::sycl::queue >( queue );
      return;
   }

   void direct_memory_manager::set_maximum_capacity( std::size_t sizeInBytes,
                                                     int device ) {

      // Retrieve the device, and make sure that it was set up already.
      device_memory& mem = get_device_memory( device );

      // Check that the request can be fulfilled.
      using info = cl::sycl::info::device;
      if( mem.m_queue->get_device().get_info< info::max_mem_alloc_size >() <
          sizeInBytes ) {
         throw std::bad_alloc();
      }
      return;
   }

   std::size_t direct_memory_manager::available_memory( int device ) const {

      // Get the object responsible for this device.
      const device_memory& mem = get_device_memory( device );

      // Get the information directly from SYCL.
      using info = cl::sycl::info::device;
      return mem.m_queue->get_device().get_info< info::max_mem_alloc_size >();
   }

   void* direct_memory_manager::allocate( std::size_t sizeInBytes,
                                          int device ) {

      // Get the object responsible for this device.
      device_memory& mem = get_device_memory( device );

      // Do the allocation.
      void* result = nullptr;
      switch( m_type ) {
      case memory_type::device:
         result = cl::sycl::malloc_device( sizeInBytes, *( mem.m_queue ) );
         break;
      case memory_type::host:
         result = cl::sycl::malloc_host( sizeInBytes, *( mem.m_queue ) );
         break;
      case memory_type::shared:
         result = cl::sycl::malloc_shared( sizeInBytes, *( mem.m_queue ) );
         break;
      default:
         assert( false );
         break;
      }

      // Update the internal state of the memory manager.
      mem.m_ptrs.push_back( result );

      // Apparently everything is okay.
      return result;
   }

   void direct_memory_manager::deallocate( void* ptr ) {

      // Find which device this allocation was made on.
      device_memory* mem = nullptr;
      if( std::find( m_defaultMemory.m_ptrs.begin(),
                     m_defaultMemory.m_ptrs.end(), ptr ) !=
          m_defaultMemory.m_ptrs.end() ) {
         mem = &m_defaultMemory;
      } else {
         auto itr = std::find_if( m_memory.begin(), m_memory.end(),
                                  [ ptr ]( const device_memory& m ) {
                                     return ( std::find( m.m_ptrs.begin(),
                                                         m.m_ptrs.end(),
                                                         ptr ) !=
                                              m.m_ptrs.end() );
                                  } );
         if( itr == m_memory.end() ) {
            throw std::runtime_error( "Couldn't find allocation" );
         }
         mem = &*itr;
      }

      // De-allocate the memory.
      cl::sycl::free( ptr, *( mem->m_queue ) );

      // Forget about this allocation.
      auto ptr_itr = std::find( mem->m_ptrs.begin(), mem->m_ptrs.end(), ptr );
      if( ptr_itr == mem->m_ptrs.end() ) {
         throw std::runtime_error( "Internal logic error detected" );
      }
      mem->m_ptrs.erase( ptr_itr );
      return;
   }

   void direct_memory_manager::reset( int device ) {

      // Get the object responsible for this device.
      device_memory& mem = get_device_memory( device );

      // Deallocate all memory associated with the device.
      for( void* ptr : mem.m_ptrs ) {
         cl::sycl::free( ptr, *( mem.m_queue ) );
      }
      mem.m_ptrs.clear();
      return;
   }

   bool direct_memory_manager::is_host_accessible() const {

      return ( m_type != memory_type::device );
   }

   const direct_memory_manager::device_memory&
   direct_memory_manager::get_device_memory( int device ) const {

      if( device == DEFAULT_DEVICE ) {
         return m_defaultMemory;
      }
      const device_memory& result = m_memory.at( device );
      if( ! result.m_queue ) {
         throw std::runtime_error( "Device ID " + std::to_string( device ) +
                                   " not set up yet" );
      }
      return result;
   }

   direct_memory_manager::device_memory&
   direct_memory_manager::get_device_memory( int device ) {

      if( device == DEFAULT_DEVICE ) {
         return m_defaultMemory;
      }
      if( static_cast< std::size_t >( device ) >= m_memory.size() ) {
         m_memory.resize( device + 1 );
      }
      device_memory& result = m_memory[ device ];
      if( ! result.m_queue ) {
         throw std::runtime_error( "Device ID " + std::to_string( device ) +
                                   " not set up yet" );
      }
      return result;
   }

   direct_memory_manager::device_memory::device_memory() {}

   direct_memory_manager::device_memory::
   device_memory( const device_memory& parent )
   : m_queue( new cl::sycl::queue( *( parent.m_queue ) ) ), m_ptrs() {

   }

   direct_memory_manager::device_memory::device_memory( device_memory&& parent )
   : m_queue( std::move( parent.m_queue ) ),
     m_ptrs( std::move( parent.m_ptrs ) ) {

   }

   direct_memory_manager::device_memory::~device_memory() {

      for( void* ptr : m_ptrs ) {
         cl::sycl::free( ptr, *m_queue );
      }
   }

   direct_memory_manager::device_memory&
   direct_memory_manager::device_memory::operator=( const device_memory& rhs ) {

      // Prevent self-assignment.
      if( this == &rhs ) {
         return *this;
      }

      // Delete all existing allocations.
      for( void* ptr : m_ptrs ) {
         cl::sycl::free( ptr, *m_queue );
      }
      m_ptrs.clear();

      // Copy the other object's queue.
      m_queue = std::make_unique< cl::sycl::queue >( *( rhs.m_queue ) );

      // Return this object.
      return *this;
   }

   direct_memory_manager::device_memory&
   direct_memory_manager::device_memory::operator=( device_memory&& rhs ) {

      // Prevent self-assignment.
      if( this == &rhs ) {
         return *this;
      }

      // Swap the internals of the two objects.
      m_queue.swap( rhs.m_queue );
      m_ptrs.swap( rhs.m_ptrs );

      // Return this object.
      return *this;
   }

} // namespace vecmem::sycl
