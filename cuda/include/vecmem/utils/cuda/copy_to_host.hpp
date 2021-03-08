/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/details/owning_vector_data.hpp"
#include "vecmem/containers/details/vector_data.hpp"
#include "vecmem/memory/resources/memory_resource.hpp"

// System include(s).
#include <type_traits>

namespace vecmem::cuda {

   /// Helper function for copying the contents of a 1-dimensional array
   template< typename TYPE >
   vecmem::details::owning_vector_data< typename std::remove_cv< TYPE >::type >
   copy_to_host( const vecmem::details::vector_data< TYPE >& device,
                 memory_resource& resource );

   /// Helper function for copying the contents of a 1-dimensional array
   template< typename TYPE >
   void copy_to_host( const vecmem::details::vector_data< TYPE >& device,
                      vecmem::details::vector_data<
                         typename std::remove_cv< TYPE >::type >& host );

} // namespace vecmem::cuda

// Include the implementation.
#include "vecmem/utils/cuda/copy_to_host.ipp"
