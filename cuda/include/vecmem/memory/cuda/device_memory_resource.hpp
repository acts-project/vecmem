/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/memory/memory_resource.hpp"

namespace vecmem::cuda {
    /**
     * @brief Memory resource that wraps direct allocations on a CUDA device.
     *
     * This is an allocator-type memory resource (that is to say, it only
     * allocates, it does not try to manage memory in a smart way) that works
     * for CUDA device memory. Each instance is bound to a specific device.
     */
    class device_memory_resource : public memory_resource {
    public:
        /**
         * @brief Construct a CUDA device resource for a specific device.
         *
         * This constructor takes a device identifier argument which works in
         * the same way as in standard CUDA code. If the device number is
         * positive, that device is selected. If the device number is negative,
         * the currently selected device is used.
         *
         * @note The default device is resolved at resource construction time,
         * not at allocation time.
         */
        device_memory_resource(int device=-1);
    private:
        virtual void * do_allocate(
            std::size_t,
            std::size_t
        ) override;

        virtual void do_deallocate(
            void * p,
            std::size_t,
            std::size_t
        ) override;

        virtual bool do_is_equal(
            const memory_resource &
        ) const noexcept override;

        const int m_device;
    };
}