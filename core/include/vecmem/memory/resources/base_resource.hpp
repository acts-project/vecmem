#pragma once

#include "vecmem/memory/resources/memory_resource.hpp"

namespace vecmem::memory::resources {
    class base_resource : public memory_resource {
    public:
        virtual bool is_host_accessible() const = 0;
    };
}