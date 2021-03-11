/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/memory/memory_resource.hpp"

#include <cstddef>

namespace vecmem {
    /**
     * @brief Downstream allocator that ensures that allocations are contiguous.
     *
     * When programming for co-processors, it is often desriable to keep
     * allocations contiguous. This downstream allocator fills that need. When
     * configured with an upstream memory resource, it will start out by
     * allocating a single, large, chunk of memory from the upstream. Then, it
     * will hand out pointers along that memory in a contiguous fashion. This
     * allocator guarantees that each consecutive allocation will start right at
     * the end of the previous.
     *
     * @note The allocation size on the upstream allocator is also the maximum
     * amount of memory that can be allocated from the contiguous memory
     * resource.
     */
    class contiguous_memory_resource : public memory_resource {
    public:
        /**
         * @brief Constructs the contiguous memory resource.
         *
         * @param[in] upstream The upstream memory resource to use.
         * @param[in] size The size of memory to allocate upstream.
         */
        contiguous_memory_resource(
            memory_resource & upstream,
            std::size_t size
        );

        /**
         * @brief Deconstruct the contiguous memory resource.
         *
         * This method deallocates the arena memory on the upstream allocator.
         */
        ~contiguous_memory_resource();
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

        memory_resource & m_upstream;
        const std::size_t m_size;
        void * const m_begin;
        void * m_next;
    };
}
