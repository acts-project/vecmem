/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/cuda/direct_memory_manager.hpp"
#include "vecmem/utils/cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

// System include(s).
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace vecmem::cuda {

   direct_memory_manager::direct_memory_manager( memory_type type,
                                                 std::size_t sizeInBytes )
   : m_type( type ) {

      // Allocate the requested amount of memory.
      set_maximum_capacity( sizeInBytes, DEFAULT_DEVICE );
   }

   direct_memory_manager::~direct_memory_manager() {

      // Free all the (still available) allocated memory.
      if( m_type == memory_type::host ) {
         for( device_memory& dev : m_memory ) {
            for( void* ptr : dev.m_ptrs ) {
               VECMEM_CUDA_ERROR_CHECK( cudaFree( ptr ) );
            }
         }
      } else {
         for( device_memory& dev : m_memory ) {
            for( void* ptr : dev.m_ptrs ) {
               VECMEM_CUDA_ERROR_CHECK( cudaFreeHost( ptr ) );
            }
         }
      }
   }

   void direct_memory_manager::set_maximum_capacity( std::size_t sizeInBytes,
                                                     int device ) {

      // Get the object responsible for this device.
      device_memory& mem = get_device_memory( device );

      // Make sure that this is possible.
      cudaDeviceProp prop;
      VECMEM_CUDA_ERROR_CHECK( cudaGetDeviceProperties( &prop, device ) );
      if( prop.totalGlobalMem < sizeInBytes ) {
         throw std::bad_alloc();
      }
      return;
   }

   std::size_t direct_memory_manager::available_memory( int device ) const {

      // Get a valid device.
      get_device( device );

      // Get the information directly from CUDA.
      cudaDeviceProp prop;
      VECMEM_CUDA_ERROR_CHECK( cudaGetDeviceProperties( &prop, device ) );
      return prop.totalGlobalMem;
    }

   void* direct_memory_manager::allocate( std::size_t sizeInBytes,
                                          int device ) {

      // Get the object responsible for this device.
      device_memory& mem = get_device_memory( device );

      // Do the allocation.
      void* result = nullptr;
      VECMEM_CUDA_ERROR_CHECK( cudaSetDevice( device ) );
      switch( m_type ) {
      case memory_type::device:
         VECMEM_CUDA_ERROR_CHECK( cudaMalloc( &result, sizeInBytes ) );
         break;
      case memory_type::host:
         VECMEM_CUDA_ERROR_CHECK( cudaMallocHost( &result, sizeInBytes ) );
         break;
      case memory_type::managed:
         VECMEM_CUDA_ERROR_CHECK( cudaMallocManaged( &result, sizeInBytes ) );
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
      auto itr = std::find_if( m_memory.begin(), m_memory.end(),
                               [ ptr ]( const device_memory& m ) {
                                  return ( std::find( m.m_ptrs.begin(),
                                                      m.m_ptrs.end(),
                                                      ptr ) != m.m_ptrs.end() );
                               } );
      if( itr == m_memory.end() ) {
         throw std::runtime_error( "Couldn't find allocation" );
      }

      // De-allocate the memory.
      if( m_type == memory_type::host ) {
         VECMEM_CUDA_ERROR_CHECK( cudaFreeHost( ptr ) );
      } else {
         VECMEM_CUDA_ERROR_CHECK( cudaFree( ptr ) );
      }

      // Forget about this allocation.
      auto ptr_itr = std::find( itr->m_ptrs.begin(), itr->m_ptrs.end(), ptr );
      if( ptr_itr == itr->m_ptrs.end() ) {
         throw std::runtime_error( "Internal logic error detected" );
      }
      itr->m_ptrs.erase( ptr_itr );
      return;
   }

   void direct_memory_manager::reset( int device ) {

      // Get the object responsible for this device.
      device_memory& mem = get_device_memory( device );

      // Deallocate all memory associated with the device.
      if( m_type == memory_type::host ) {
         for( void* ptr : mem.m_ptrs ) {
            VECMEM_CUDA_ERROR_CHECK( cudaFreeHost( ptr ) );
         }
      } else {
         for( void* ptr : mem.m_ptrs ) {
            VECMEM_CUDA_ERROR_CHECK( cudaFree( ptr ) );
         }
      }
      mem.m_ptrs.clear();
      return;
   }

   void direct_memory_manager::get_device( int& device ) {

      // If the user didn't ask for a specific device, use the one currently
      // used by CUDA.
      if( device == DEFAULT_DEVICE ) {
         VECMEM_CUDA_ERROR_CHECK( cudaGetDevice( &device ) );
      }
      return;
   }

   direct_memory_manager::device_memory&
   direct_memory_manager::get_device_memory( int& device ) {

      // Get a valid device.
      get_device( device );

      // Make sure that the internal storage variable is large enough.
      if( static_cast< std::size_t >( device ) >= m_memory.size() ) {
         m_memory.resize( device + 1 );
      }

      // Return the requested object.
      return m_memory[ device ];
   }

} // namespace vecmem::cuda
