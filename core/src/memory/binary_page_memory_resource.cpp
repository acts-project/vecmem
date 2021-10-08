/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/binary_page_memory_resource.hpp"

#include <algorithm>
#include <memory>
#include <stack>

#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/debug.hpp"

namespace {
/**
 * @brief Rounds a size up to the nearest power of two.
 */
std::size_t round_up(std::size_t size) {
    for (unsigned short i = 0; i <= 32; i++) {
        std::size_t s = 2UL << i;

        if (s >= size) {
            return s;
        }
    }

    return 0;
}
}  // namespace

namespace vecmem {
binary_page_memory_resource::binary_page_memory_resource(
    memory_resource &upstream)
    : m_upstream(upstream) {}

binary_page_memory_resource::~binary_page_memory_resource() {
    /*
     * We only need to deallocate the root pages here.
     */
    for (std::unique_ptr<page> &p : m_pages) {
        m_upstream.deallocate(p->addr, p->size);
    }
}

void *binary_page_memory_resource::do_allocate(std::size_t size, std::size_t) {
    VECMEM_DEBUG_MSG(5, "Request received for %ld bytes", size);
    /*
     * First, we round our allocation request up to a power of two, since
     * that is what the sizes of all our pages are.
     */
    std::size_t goal = round_up(size);
    VECMEM_DEBUG_MSG(5, "Will be allocating %ld bytes instead", goal);

    /*
     * Attempt to find a free page that can fit our allocation goal.
     */
    page *cand = find_free_page(goal);

    /*
     * If we don't have a candidate, there is no available page that can fit
     * our request. First, we allocate a new root page from the upstream
     * allocator, and then look for that new page.
     */
    if (cand == nullptr) {
        allocate_upstream(goal);

        cand = find_free_page(goal);
    }

    /*
     * If there is still no candidate, something has gone wrong and we
     * cannot recover.
     */
    if (cand == nullptr) {
        throw std::bad_alloc();
    }

    /*
     * If the page is split (but its children are all free), we will first
     * need to unsplit it.
     */
    if (cand->state == page_state::SPLIT) {
        cand->unsplit();
    }

    /*
     * Keep splitting the page until we have reached our target size.
     */
    while (cand->size > goal) {
        cand->split();
        cand = cand->left.get();
    }

    /*
     * Mark the page as occupied, then return the address.
     */
    cand->state = page_state::OCCUPIED;

    VECMEM_DEBUG_MSG(2, "Allocated %ld (%ld) bytes at %p", size, goal,
                     cand->addr);
    return cand->addr;
}

void binary_page_memory_resource::do_deallocate(void *p, std::size_t,
                                                std::size_t) {

    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p", p);

    page *cand = nullptr;

    /*
     * We will use this stack to perform a depth-first search of the pages
     * in our binary trees.
     *
     * TODO: This would be much more efficient with a hashmap.
     */
    std::stack<page *> rem;

    for (std::unique_ptr<page> &pg : m_pages) {
        rem.push(pg.get());
    }

    /*
     * Iterate over the pages in our tree.
     */
    while (!rem.empty()) {
        page *c = rem.top();
        rem.pop();

        if (c->addr == p && c->state != page_state::SPLIT) {
            /*
             * If we have found the target node, we're done.
             */
            cand = c;
            break;
        } else if (c->state == page_state::SPLIT) {
            /*
             * If the page is split, we need to add its children to the
             * stack.
             */
            rem.push(c->left.get());
            rem.push(c->right.get());
        }
    }

    /*
     * If we have found the target, just mark it as vacant. There is no need
     * to issue a deallocation upstream.
     */
    if (cand != nullptr) {
        cand->free();
    }
}

binary_page_memory_resource::page *binary_page_memory_resource::find_free_page(
    std::size_t size) {
    page *cand = nullptr;

    /*
     * Here we also do a depth-first search, looking through all our pages
     * to find the smallest free page that can fit our request.
     */
    std::stack<page *> rem;

    for (std::unique_ptr<page> &p : m_pages) {
        rem.push(p.get());
    }

    /*
     * Core DFS loop.
     */
    while (!rem.empty()) {
        page *c = rem.top();
        rem.pop();

        /*
         * For split pages, we will need to examine both children as part of
         * our search.
         */
        if (c->state == page_state::SPLIT) {
            rem.push(c->left.get());
            rem.push(c->right.get());
        }

        /*
         * A page is a candidate if it's free, if it's big enough, and if
         * it's smaller than any previous candidate.
         */
        if (c->is_free() && c->size >= size &&
            (cand == nullptr || c->size < cand->size)) {
            cand = c;
        }

        /*
         * If the size of the page is exactly equal to the size of our
         * rounded request, we will never find a smaller page and we can
         * safely return what we have.
         */
        if (cand != nullptr && cand->size == size) {
            break;
        }
    }

    return cand;
}

void binary_page_memory_resource::allocate_upstream(std::size_t size) {
    /*
     * Making too many small allocations here would be a bad idea, so we
     * take a minimum size of 1 megabyte.
     */
    size = std::max(size, static_cast<std::size_t>(1048576));

    /*
     * Allocate the memory upstream and gather the address.
     */
    void *addr = m_upstream.allocate(size);

    /*
     * Create a new page and add the information we have about it.
     */
    std::unique_ptr<page> newp = std::make_unique<page>();

    newp->state = page_state::VACANT;
    newp->size = size;
    newp->addr = addr;

    /*
     * Add our new page to the list of root pages.
     */
    m_pages.push_back(std::move(newp));
}

bool binary_page_memory_resource::page::is_free() {
    if (state == page_state::VACANT) {
        /*
         * Vacant pages are always free.
         */
        return true;
    } else if (state == page_state::SPLIT) {
        /*
         * Split pages are free if and only if both of their children are
         * free.
         */
        return left->is_free() && right->is_free();
    } else {
        /*
         * Occupied pages are never free.
         */
        return false;
    }
}

void binary_page_memory_resource::page::free() {
    if (state == page_state::OCCUPIED) {
        /*
         * To free an occupied page, just mark it as vacant.
         */
        state = page_state::VACANT;
    } else if (state == page_state::SPLIT) {
        /*
         * To free a split page, we recursively free its children, but we do
         * not change it's state to vacant, nor do we delete the children.
         */
        left->free();
        right->free();
    }
}

void binary_page_memory_resource::page::split() {
    /*
     * Splitting a non-vacant page is an unrecoverable error.
     */
    if (state != page_state::VACANT) {
        throw std::runtime_error("Can't split a non-vacant page!");
    }

    /*
     * Mark the page as split.
     */
    state = page_state::SPLIT;

    /*
     * Create the left child, starting out as vacant, with size half of its
     * parent.
     */
    left = std::make_unique<page>();
    left->state = page_state::VACANT;
    left->size = size / 2;
    left->addr = addr;

    /*
     * Create the right child in much the same way as the left child, except
     * the starting address is halfway through the parent page.
     */
    right = std::make_unique<page>();
    right->state = page_state::VACANT;
    right->size = size / 2;
    right->addr =
        static_cast<void *>(static_cast<char *>(left->addr) + left->size);
}

void binary_page_memory_resource::page::unsplit() {
    /*
     * If the page is not split, we can't unsplit it for obvious reasons.
     */
    if (state != page_state::SPLIT) {
        return;
    }

    /*
     * If the children are not vacant, merging would be potentially
     * disastrous, so we cannot continue.
     */
    if (left->state != page_state::VACANT ||
        right->state != page_state::VACANT) {
        return;
    }

    /*
     * If everything is in order, deallocate the children and mark the page
     * as vacant.
     */
    left.reset(nullptr);
    right.reset(nullptr);

    state = page_state::VACANT;
}
}  // namespace vecmem
