/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/array.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/cuda/copy.hpp"
#include "test_cuda_containers_kernels.cuh"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test fixture for the on-device vecmem container tests
class cuda_containers_test : public testing::Test {};

/// Test a linear transformation using the managed memory resource
TEST_F( cuda_containers_test, managed_memory ) {

   // The managed memory resource.
   vecmem::cuda::managed_memory_resource managed_resource;

   // Create an input and an output vector in managed memory.
   vecmem::vector< int > inputvec( { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                                   &managed_resource );
   vecmem::vector< int > outputvec( inputvec.size(), &managed_resource );
   EXPECT_EQ( inputvec.size(), outputvec.size() );

   // Create the array that is used in the linear transformation.
   vecmem::array< int, 2 > constants( managed_resource );
   constants[ 0 ] = 2;
   constants[ 1 ] = 3;

   // Perform a linear transformation using the vecmem vector helper types.
   linearTransform( vecmem::get_data( constants ),
                    vecmem::get_data( inputvec ),
                    vecmem::get_data( outputvec ) );

   // Check the output.
   EXPECT_EQ( inputvec.size(), outputvec.size() );
   for( std::size_t i = 0; i < outputvec.size(); ++i ) {
      EXPECT_EQ( outputvec.at( i ),
                 inputvec.at( i ) * constants[ 0 ] + constants[ 1 ] );
   }
}

/// Test a linear transformation while hand-managing the memory copies
TEST_F( cuda_containers_test, explicit_memory ) {

   // The host/device memory resources.
   vecmem::cuda::device_memory_resource device_resource;
   vecmem::cuda::host_memory_resource host_resource;

   // Helper object for performing memory copies.
   vecmem::cuda::copy copy;

   // Create input/output vectors on the host.
   vecmem::vector< int > inputvec( { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                                   &host_resource );
   vecmem::vector< int > outputvec( inputvec.size(), &host_resource );
   EXPECT_EQ( inputvec.size(), outputvec.size() );

   // Allocate a device memory block for the output container.
   auto outputvechost = vecmem::get_data( outputvec );
   vecmem::data::vector_buffer< int >
      outputvecdevice( outputvec.size(), device_resource );

   // Create the array that is used in the linear transformation.
   vecmem::array< int, 2 > constants( host_resource );
   constants[ 0 ] = 2;
   constants[ 1 ] = 3;

   // Perform a linear transformation with explicit memory copies.
   linearTransform( copy.to( vecmem::get_data( constants ), device_resource,
                             vecmem::copy::type::host_to_device ),
                    copy.to( vecmem::get_data( inputvec ), device_resource ),
                    outputvecdevice );
   copy( outputvecdevice, outputvechost, vecmem::copy::type::device_to_host );

   // Check the output.
   EXPECT_EQ( inputvec.size(), outputvec.size() );
   for( std::size_t i = 0; i < outputvec.size(); ++i ) {
      EXPECT_EQ( outputvec.at( i ),
                 inputvec.at( i ) * constants[ 0 ] + constants[ 1 ] );
   }
}
