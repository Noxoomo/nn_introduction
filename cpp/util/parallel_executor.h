#pragma once

#include "singleton.h"
#include "semaphore.h"

#include <cstdint>
#include <functional>
#include <utility>

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

template <int N>
inline ThreadPool& GlobalThreadPool() {
    return Singleton<N, ThreadPool>();
}

template <class Task>
inline void parallelForInThreadPool(ThreadPool& pool, int64_t from, int64_t to, Task&& task, bool parallel = true,
                                    decltype(std::declval<Task>()(1, 1))* unused = NULL) {
    const int64_t numBlocks = pool.numThreads();
    const int64_t blockSize = (to - from + numBlocks - 1) / numBlocks;

    Semaphore sema;
    SemaphoreAcquireGuard sag(sema, to - from);

    for (int64_t blockId = 0; blockId < numBlocks; ++blockId) {
        const int64_t startBlock = std::min<int64_t>(blockId * blockSize, to);
        const int64_t endBlock = std::min<int64_t>((blockId + 1) * blockSize, to);
        if (startBlock != endBlock) {
            if (parallel) {
                pool.enqueue([startBlock, endBlock, blockId, &task, &sema] {
                    SemaphoreReleaseGuard srg(sema, endBlock - startBlock);
                    for (int64_t i = startBlock; i < endBlock; ++i) {
                        task(blockId, i);
                    }
                });
            } else {
                for (int64_t i = startBlock; i < endBlock; ++i) {
                    task(blockId, i);
                }
            }
        }
    }
}

template <class Task>
inline void parallelForInThreadPool(ThreadPool& pool, int64_t from, int64_t to, Task&& task, bool parallel = true,
                                    decltype(std::declval<Task>()(1))* unused = NULL) {
    const int64_t numBlocks = pool.numThreads();
    const int64_t blockSize = (to - from + numBlocks - 1) / numBlocks;

    Semaphore sema;
    SemaphoreAcquireGuard sag(sema, to - from);

    for (int64_t blockId = 0; blockId < numBlocks; ++blockId) {
        const int64_t startBlock = std::min<int64_t>(blockId * blockSize, to);
        const int64_t endBlock = std::min<int64_t>((blockId + 1) * blockSize, to);
        if (startBlock != endBlock) {
            if (parallel) {
                pool.enqueue([startBlock, endBlock, blockId, &task, &sema] {
                    SemaphoreReleaseGuard srg(sema, endBlock - startBlock);
                    for (int64_t i = startBlock; i < endBlock; ++i) {
                        task(i);
                    }
                });
            } else {
                for (int64_t i = startBlock; i < endBlock; ++i) {
                    task(i);
                }
            }
        }
    }
}

template <class Task>
inline void parallelFor(int64_t from, int64_t to, Task&& task, bool parallel = true) {
    auto& pool = GlobalThreadPool<0>();
    parallelForInThreadPool(pool, from, to, task, parallel);
}

template <int N, class Task>
inline void parallelFor(int64_t from, int64_t to, Task&& task, bool parallel = true) {
    auto& pool = GlobalThreadPool<N>();
    parallelForInThreadPool(pool, from, to, task, parallel);
}