#pragma once

#include <memory_resource>

#include "vecmem/memory/resources/base_resource.hpp"

namespace vecmem::memory {
    template<typename T>
    class polymorphic_allocator {
    public:
        typedef T value_type;

        typedef std::false_type propagate_on_container_move_assignment;
        typedef std::false_type is_always_equal;

        polymorphic_allocator(
            resources::base_resource * res
        ) : res(res), alloc(res) {};

        T * allocate(std::size_t size) {
            return alloc.allocate(size);
        };

        void deallocate(T * ptr, std::size_t size) {
            alloc.deallocate(ptr, size);
        };

        template<typename U, typename... Args>
        void construct(U* ptr, Args&&... args) {
            if (res->is_host_accessible()) {
                alloc.construct(ptr, std::forward<Args>(args)...);
            }
        };

        template<typename U>
        void destroy(U* ptr) {
            if (res->is_host_accessible()) {
                alloc.destroy(ptr);
            }
        };
    private:
        resources::base_resource * res;
        std::pmr::polymorphic_allocator<T> alloc;
    };
}