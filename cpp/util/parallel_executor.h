#pragma once

#include "singleton.h"
#include <cstdint>
#include <functional>

#include <c10/core/thread_pool.h>
#include <thread>

class ThreadPool {
public:
    ThreadPool();

    template <class Task>
    void enqueue(Task&& task) {
        pool_.run(std::forward<Task>(task));
    }

    void waitComplete() {
        pool_.waitWorkComplete();
    }

    int64_t numThreads() const {
        return pool_.size();
    }
private:

    c10::ThreadPool pool_;
};


inline ThreadPool& GlobalThreadPool() {
    return Singleton<ThreadPool>();
}


template <class Task>
inline void parallelFor(int64_t from, int64_t to, Task&& task) {
    auto& pool = GlobalThreadPool();
    const int64_t numBlocks = pool.numThreads();
    const int64_t blockSize = (to - from + numBlocks - 1) / numBlocks;
    for (int64_t blockId = 0; blockId < numBlocks; ++blockId) {
        const int64_t startBlock = std::min<int64_t>(blockId * blockSize, to);
        const int64_t endBlock = std::min<int64_t>((blockId + 1) * blockSize, to);
        if (startBlock != endBlock) {
            pool.enqueue([startBlock, endBlock, &task] {
                for (int64_t i = startBlock; i < endBlock; ++i) {
                    task(i);
                }
            });
        }
    }
    pool.waitComplete();
}
