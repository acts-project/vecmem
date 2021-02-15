#pragma once

#if __has_include(<memory_resource>)
#include <memory_resource>

namespace vecmem::memory::resources {
    using memory_resource = std::pmr::memory_resource;

    template<typename T>
    using polymorphic_allocator = std::pmr::polymorphic_allocator<T>;
}
#else
#include <experimental/memory_resource>

namespace vecmem::memory::resources {
    using memory_resource = std::experimental::pmr::memory_resource;

    template<typename T>
    using polymorphic_allocator = std::experimental::pmr::polymorphic_allocator<T>;
}
#endif