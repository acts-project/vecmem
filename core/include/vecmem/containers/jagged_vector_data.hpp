/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/containers/details/jagged_vector_view.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/vector.hpp"

#include <cstddef>

namespace vecmem {
    template<typename T>
    class jagged_vector_view;

    /**
     * @brief A data wrapper for jagged vectors.
     *
     * This class constructs the relevant administrative data from a vector of
     * vectors, and is designed to be later turned into a @c jagged_vector_view
     * object.
     */
    template<typename T>
    class jagged_vector_data : public details::jagged_vector_view<T> {
    public:
        using base_type = details::jagged_vector_view<T>;

        /**
         * @brief Construct jagged vector data from a jagged vector.
         *
         * This class converts from std vectors (or rather, vecmem::vectors) to
         * a jagged vector data.
         *
         * @param[in] vec The jagged vector to make a data view for.
         * @param[in] mem The memory resource to manage the internal state. If
         * set to nullptr, uses the same memory resource as the jagged vector.
         */
        jagged_vector_data(
            jagged_vector<T> & vec,
            memory_resource * mem = nullptr
        );

        /**
         * @brief Destruct the jagged vector data.
         *
         * This destructor does not affect the viewed data in any way. The
         * internal state is destroyed if and only if this object is not a copy.
         */
        ~jagged_vector_data(
            void
        );

    private:
        /**
         * The memory manager used to manage the internal state (row data) of
         * the jagged data.
         */
        memory_resource * m_mem;
    };
}

#include "jagged_vector_data.ipp"
