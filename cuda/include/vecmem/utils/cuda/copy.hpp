/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_buffer.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/memory/memory_resource.hpp"

namespace vecmem::cuda {

   /// @name 1-dimensional vector data handling functions
   /// @{

   /// Function copying a buffer from the host to the device
   template< typename TYPE >
   vecmem::data::vector_buffer< TYPE >
   copy_to_device( const vecmem::data::vector_view< TYPE >& host,
                   memory_resource& resource );

   /// Function copying a buffer from the device to the host
   template< typename TYPE >
   vecmem::data::vector_buffer< TYPE >
   copy_to_host( const vecmem::data::vector_view< TYPE >& device,
                 memory_resource& resource );

   /// Helper function for copying the contents of a 1-dimensional array
   template< typename TYPE >
   void copy( const vecmem::data::vector_view< TYPE >& from,
              vecmem::data::vector_view< TYPE >& to );

   /// @}

   /// @name Jagged vector data handling functions
   /// @{

   /// Function copying the internal state of a jagged vector buffer
   template< typename TYPE >
   void prepare_for_device( vecmem::data::jagged_vector_buffer< TYPE >& data );

   /// Function copying a jagged vector's data from the host to the device
   template< typename TYPE >
   vecmem::data::jagged_vector_buffer< TYPE >
   copy_to_device( const vecmem::data::jagged_vector_view< TYPE >& host,
                   memory_resource& device_resource,
                   memory_resource& host_resource );

   /// Function copying a jagged vector's data from the device to the host
   template< typename TYPE >
   vecmem::data::jagged_vector_buffer< TYPE >
   copy_to_host( const vecmem::data::jagged_vector_buffer< TYPE >& device,
                 memory_resource& host_resource );

   /// Helper function copying the contents of two jagged arrays/vectors
   template< typename TYPE >
   void copy( const vecmem::data::jagged_vector_view< TYPE >& from,
              vecmem::data::jagged_vector_view< TYPE >& to );

   /// Helper function copying the contents of two jagged arrays/vectors
   template< typename TYPE >
   void copy( const vecmem::data::jagged_vector_buffer< TYPE >& from,
              vecmem::data::jagged_vector_view< TYPE >& to );

   /// Helper function copying the contents of two jagged arrays/vectors
   template< typename TYPE >
   void copy( const vecmem::data::jagged_vector_view< TYPE >& from,
              vecmem::data::jagged_vector_buffer< TYPE >& to );

   /// Helper function copying the contents of two jagged arrays/vectors
   template< typename TYPE >
   void copy( const vecmem::data::jagged_vector_buffer< TYPE >& from,
              vecmem::data::jagged_vector_buffer< TYPE >& to );

   /// @}

} // namespace vecmem::cuda

// Include the implementation.
#include "vecmem/utils/cuda/impl/copy.ipp"
