#pragma once
#include <memory>

//the most straightforward way...
namespace Private {

    template <class T, int N = 0>
    class SingletonImpl {
    public:
        T* operator->() {
            return instance();
        }

        const T* operator->() const {
            return instance();
        }

        T& operator*() {
            return *instance();
        }

        const T& operator*() const {
            return *instance();
        }
    private:

        T* instance() const {
            static T instance_;
            return &instance_;
        }

        template <class TC>
        friend TC& Instance();
    };

    template <class T, int N = 0>
    inline T& Instance() {
        return *Private::SingletonImpl<T, N>();
    }

    template <class T, int N = 0>
    class TlsSingletonImpl {
    public:
        T* operator->() {
            return instance();
        }

        const T* operator->() const {
            return instance();
        }

        T& operator*() {
            return *instance();
        }

        const T& operator*() const {
            return *instance();
        }
    private:

        T* instance() const {
            thread_local T instance_;
            return &instance_;
        }

        template <class TC>
        friend TC& TlsInstance();
    };

    template <class T, int N = 0>
    inline T& TlsInstance() {
        return *Private::TlsSingletonImpl<T, N>();
    }
}

template <class T, int N = 0>
inline T& Singleton() {
    return Private::Instance<T, N>();
};

template <class T, int N = 0>
inline T& TlsSingleton() {
    return Private::TlsInstance<T>();
};
