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
#include "vecmem/vecmem_vitis_export.hpp"

#include <CL/cl.hpp>
#include <CL/cl_ext_xilinx.h>

namespace vecmem::vitis {

/// Specialisation of @c vecmem::copy for Vitis
class VECMEM_VITIS_EXPORT copy : public vecmem::copy {

protected:
    /// Perform a memory copy using CUDA
    virtual void do_copy(std::size_t size, const void* from, void* to,
                         type::copy_type cptype) const override final;
    /// Fill a memory area using CUDA
    virtual void do_memset(std::size_t size, void* ptr,
                           int value) const override final;

};  // class copy

}  // namespace vecmem::vitis
