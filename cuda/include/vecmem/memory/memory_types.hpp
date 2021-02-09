/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem::memory {
   enum class memory_type {
      //! Memory allocated in device memory
      DEVICE,
      //! Pinned memory allocated on the host
      HOST,
      //! CUDA managed memory allocated on the device and host
      MANAGED
   };
}
