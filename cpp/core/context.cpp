#include "context.h"
#include <util/singleton.h>
#include <util/guard.h>
#include <c10/core/TensorOptions.h>
#include <util/parallel_executor.h>

//TODO(noxoomo): i think context should be thread-local
namespace {

    class Context {
    public:
        const ComputeDevice& device() const {
            return current_;
        }

        void setDevice(const ComputeDevice& device) {
            with_guard(guard_) {
                current_ = device;
            }
        }

    private:
        ComputeDevice current_ = ComputeDevice(ComputeDeviceType::Cpu);
        std::mutex guard_;
    };
}

const ComputeDevice& CurrentDevice() {
    return Singleton<Context>().device();
}

void SetDevice(const ComputeDevice& device) {
    Singleton<Context>().setDevice(device);
}

