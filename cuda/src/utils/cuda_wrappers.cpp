/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "cuda_error_handling.hpp"

#include <cuda_runtime_api.h>

namespace vecmem::cuda::details {
    int get_device() {
        int d;

        VECMEM_CUDA_ERROR_CHECK(cudaGetDevice(&d));

        return d;
    }
}