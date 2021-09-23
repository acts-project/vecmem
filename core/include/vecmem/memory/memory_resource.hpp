/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// This header can only be used in C++17 mode. So "device compilers" that
// can't use C++17, must not see it. Which in practically all cases should
// "just" be a question of code organisation.
#if __cplusplus < 201700L
#error \
    "This header can only be used in C++17 mode. " \
         "Ideally it should only be used by the \"host compiler\"."
#endif  // < C++17

/*
 * The purpose of this file is to provide uniform access (on a source-code
 * level) to the memory_resource type from the standard library. These are
 * either in the std::pmr namespace or in the std::experimental::pmr namespace
 * depending on the GCC version used, so we try to unify them by aliassing
 * depending on the compiler feature flags.
 */
#if defined(VECMEM_HAVE_PMR_MEMORY_RESOURCE)
#include <memory_resource>

namespace vecmem {
using memory_resource = std::pmr::memory_resource;
}
#elif defined(VECMEM_HAVE_EXPERIMENTAL_PMR_MEMORY_RESOURCE)
#include <experimental/memory_resource>

namespace vecmem {
using memory_resource = std::experimental::pmr::memory_resource;
}
#else
#error "C++17 LFTS V1 (P0220R1) component memory_resource not found!?!"
#endif

#if defined(VECMEM_NOT_HAVE_DEFAULT_RESOURCE)

#include <atomic>

#if defined(VECMEM_HAVE_PMR_MEMORY_RESOURCE)
namespace std::pmr {
#else
namespace std::experimental::pmr {
#endif

namespace detail {
class new_delete_resource_impl : public memory_resource {
    virtual void* do_allocate(std::size_t bytes, std::size_t) override {
        return malloc(bytes);
    }

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override {
        free(p);
    }

    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override {
        return &other == this;
    }
};

inline std::atomic<memory_resource*>& default_resource() noexcept {
    static std::atomic<memory_resource*> res{new_delete_resource()};
    return res;
}

}  // namespace detail

inline memory_resource* new_delete_resource() noexcept {
    static detail::new_delete_resource_impl res{};
    return &res;
}

inline memory_resource* get_default_resource() noexcept {
    return detail::default_resource();
}

inline memory_resource* set_default_resource(memory_resource* res) noexcept {

    memory_resource* new_res = res == nullptr ? new_delete_resource() : res;

    return detail::default_resource().exchange(new_res);
}
}
#endif
