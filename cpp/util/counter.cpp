#include "counter.h"
#include "singleton.h"

#include <atomic>

namespace {
    class CounterImpl {
    public:
        CounterImpl()
        : id_(0) {
        }

        int64_t next() {
            return id_.fetch_add(1);
        }

    private:
        std::atomic_int_fast64_t id_;
    };


}

int64_t Counter::next() {
    auto& impl = Singleton<CounterImpl>();
    return impl.next();
}
