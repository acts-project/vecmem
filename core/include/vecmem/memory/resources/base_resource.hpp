#pragma once

#include <memory_resource>

namespace vecmem::memory::resources {
    class base_resource : public std::pmr::memory_resource {
    public:
        virtual bool is_host_accessible() const = 0;
    };
}