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

namespace vecmem::hip {

   /// Specialisation of @c vecmem::copy for HIP
   class copy : public vecmem::copy {

   protected:
      /// Perform a memory copy using HIP
      virtual void do_copy( std::size_t size, const void* from, void* to,
                            type::copy_type cptype ) const override;

   }; // class copy

} // namespace vecmem::hip
