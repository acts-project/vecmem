/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/vector.hpp"

#include <cstddef>

namespace vecmem::data {
    template<typename T>
    jagged_vector_data<T>::jagged_vector_data(
        jagged_vector<T> & vec,
        memory_resource * mem
    ) :
        base_type(
            vec.size(),
            static_cast<vector_view<T> *>(
                (mem == nullptr ? vec.get_allocator().resource() : mem)->allocate(
                    vec.size() * sizeof(vector_view<T>)
                )
            )
        ),
        m_mem(mem == nullptr ? vec.get_allocator().resource() : mem)
    {
        /*
         * To construct a jagged view, we copy the important information (the
         * size and starting pointer) of the standard vectors to our reduced
         * complexity format.
         */
        for (std::size_t i = 0; i < base_type::m_size; ++i) {
            /*
             * We use the memory allocated earlier and construct device vector
             * objects there.
             */
            new (base_type::m_ptr + i) vector_view<T>(
                vec.at(i).size(),
                vec.at(i).data()
            );
        }
    }

    template<typename T>
    jagged_vector_data<T>::~jagged_vector_data(
        void
    ) {
        /*
         * Use the memory manager to deallocate the memory owned by this object.
         */
        m_mem->deallocate(
            base_type::m_ptr,
            base_type::m_size * sizeof(vector_view<T>)
        );
    }
} // namespace vecmem::data
