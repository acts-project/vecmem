#include <atomic>

#include "vecmem/memory/memory_resource.hpp"

#if defined(VECMEM_HAVE_PMR_MEMORY_RESOURCE)
namespace std::pmr {
#else
namespace std::experimental {
inline namespace fundamentals_v1 {
namespace pmr {
#endif

namespace {
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

std::atomic<memory_resource*>& default_resource() noexcept {
    static std::atomic<memory_resource*> res{new_delete_resource()};
    return res;
}

}  // namespace

memory_resource* new_delete_resource() noexcept {
    static new_delete_resource_impl res{};
    return &res;
}

memory_resource* get_default_resource() noexcept {
    return default_resource();
}

memory_resource* set_default_resource(memory_resource* res) noexcept {

    memory_resource* new_res = res == nullptr ? new_delete_resource() : res;

    return default_resource().exchange(new_res);
}
#if defined(VECMEM_HAVE_PMR_MEMORY_RESOURCE)
}
#else
}  // namespace pmr
}  // namespace fundamentals_v1
}  // namespace std::experimental
#endif
