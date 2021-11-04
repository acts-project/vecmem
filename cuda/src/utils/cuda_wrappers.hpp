/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

namespace vecmem::cuda::details {
/**
 * @brief Get current CUDA device number.
 *
 * This function wraps the cudaGetDevice function in a way that returns the
 * device number rather than use a reference argument to write to.
 *
 * Note that calling the function on a machine with no CUDA device does not
 * result in an error, the function just returns 0 in that case.
 */
int get_device();
}  // namespace vecmem::cuda::details
