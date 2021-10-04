/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
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
class VECMEM_CORE_EXPORT contiguous_memory_resource
    : public details::memory_resource_base {

public:
    /**
     * @brief Constructs the contiguous memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     * @param[in] size The size of memory to allocate upstream.
     */
    contiguous_memory_resource(memory_resource& upstream, std::size_t size);

    /**
     * @brief Deconstruct the contiguous memory resource.
     *
     * This method deallocates the arena memory on the upstream allocator.
     */
    ~contiguous_memory_resource();

private:
    /// @name Function(s) implemented from @c vecmem::memory_resource
    /// @{

    /// Allocate memory "just after" the previous allocation
    virtual void* do_allocate(std::size_t, std::size_t) override;
    /// De-allocate a block of previously allocated memory
    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    /// @}

    /// Upstream memory resource to allocate the one memory blob with
    memory_resource& m_upstream;
    /// Size of memory to allocate upstream
    const std::size_t m_size;
    /// Pointer to the memory blob allocated from upstream
    void* const m_begin;
    /// Pointer to the next free memory block to give out
    void* m_next;

};  // class contiguous_memory_resource

}  // namespace vecmem
