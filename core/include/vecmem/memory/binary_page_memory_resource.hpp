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
#include <memory>
#include <vector>

namespace vecmem {
    /**
     * @brief A memory manager using power-of-two pages that can be split to
     * deal with allocation requests of various sizes.
     *
     * This is a non-terminal memory resource which relies on an upstream
     * allocator to do the actual allocation. The allocator will allocate only
     * large blocks with sizes power of two from the upstream allocator. These
     * blocks can then be split in half and allocated, split in half again. This
     * creates a binary tree of pages which can be either vacant, occupied, or
     * split.
     */
    class binary_page_memory_resource : public memory_resource {
    public:
        /**
         * @brief Initialize a binary page memory manager depending on an
         * upstream memory resource.
         */
        binary_page_memory_resource(memory_resource &);

        /**
         * @brief Deconstruct a binary page memory manager, freeing all
         * allocated blocks upstream.
         */
        ~binary_page_memory_resource();
    private:
        /**
         * @brief The different possible states a page can be in.
         *
         * We define three different page states. An OCCUPIED state is non-free,
         * and used directly (thus it is not split). A VACANT page is not split
         * and unused. A SPLIT page is split in two, and has two children pages.
         */
        enum class page_state {
            OCCUPIED,
            VACANT,
            SPLIT
        };

        /**
         * @brief Representation of a single page of memory.
         */
        struct page {
            /**
             * The current state of this page.
             */
            page_state state;

            /**
             * The size of this page. This should always be a power of two.
             */
            std::size_t size;

            /**
             * The starting address of this page. This is not necessarily host
             * accessible memory.
             */
            void * addr;

            /**
             * The left and right children of this page. These should be null
             * for OCCUPIED or VACANT pages, and non-null for SPLIT pages.
             */
            std::unique_ptr<page> left = nullptr, right = nullptr;

            /**
             * @brief Determine whether this page is free.
             *
             * Note that this does not only include vacant pages, but also
             * split pages in which both children are recursively also free.
             */
            bool is_free();

            /**
             * @brief Free the page.
             *
             * The behaviour of this method is dependent on the state of the
             * page. For VACANT pages, this is a no-op. For OCCUPIED pages, this
             * simply marks the page as VACANT. For SPLIT pages, the page
             * remains split, but the children are recursively freed according
             * to a similar method.
             */
            void free();

            /**
             * @brief Split a page into two equally sized pages.
             *
             * This method is used to split a page, and it populates the left
             * and right children with new pages.
             */
            void split();

            /**
             * @brief Unsplit a page, deletings its two children.
             *
             * This method only works if the page is split, and if its children
             * are vacant.
             */
            void unsplit();
        };

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

        /**
         * @brief Find the smallest free page that could fit the requested size.
         *
         * Note that this method might return split pages if both children are
         * free. In that case, the page should first be unsplit. In some cases,
         * the returned page might be (significantly) larger than the request,
         * and should be split before allocating.
         */
        page * find_free_page(
            std::size_t
        );

        /**
         * @brief Perform an upstream allocation.
         *
         * This method performs an allocation through the upstream memory
         * resource and immediately creates a page to represent this new chunk
         * of memory.
         */
        void allocate_upstream(
            std::size_t
        );

        memory_resource & m_upstream;
        std::vector<std::unique_ptr<page>> m_pages;
    };
}
