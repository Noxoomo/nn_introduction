#pragma once

#include "exception.h"

//copy-paste from torch: i don't need all functionality, but need const refs
namespace Detail {

    template <typename T>
    class ArrayRef final {
    public:
        using iterator = T*;
        using const_iterator = const T*;
        using size_type = size_t;

        using reverse_iterator = std::reverse_iterator<iterator>;

    private:
        /// The start of the array, in an external buffer.
        T* Data;
        /// The number of elements.
        size_type Length;
    public:
        /// @name Constructors
        /// @{

        /// Construct an empty ArrayRef.
        /* implicit */ constexpr ArrayRef() : Data(nullptr), Length(0) {}

        /// Construct an ArrayRef from a single element.
        // TODO Make this explicit
        constexpr ArrayRef(T& OneElt) : Data(&OneElt), Length(1) {}

        /// Construct an ArrayRef from a pointer and length.
        constexpr ArrayRef(T* data, size_t length)
            : Data(data), Length(length) {}

        /// Construct an ArrayRef from a range.
        constexpr ArrayRef(T* begin, T* end)
            : Data(begin), Length(end - begin) {}

        /// Construct an ArrayRef from a range.
        constexpr ArrayRef(std::vector<std::remove_const_t<T>>& vec)
            : Data(vec.data()), Length(vec.size()) {}

        constexpr ArrayRef(const std::vector<std::remove_const_t<T>>& vec)
            : Data(vec.data()), Length(vec.size()) {}
//
//        constexpr ArrayRef(const ArrayRef<std::remove_const_t<T>>& vec)
//            : Data(vec.data()), Length(vec.size()) {}
        /// @}
        /// @name Simple Operations
        /// @{

        constexpr ArrayRef(const ArrayRef& other)
        : Data(other.Data)
        , Length(other.Length) {

        }


        operator ArrayRef<const T>() const {
            return ArrayRef<const T>(Data, Length);
        }

        constexpr iterator begin() const {
            return Data;
        }
        constexpr iterator end() const {
            return Data + Length;
        }

        constexpr const_iterator cbegin() const {
            return Data;
        }
        constexpr const_iterator cend() const {
            return Data + Length;
        }

        constexpr reverse_iterator rbegin() const {
            return reverse_iterator(end());
        }
        constexpr reverse_iterator rend() const {
            return reverse_iterator(begin());
        }

        /// empty - Check if the array is empty.
        constexpr bool empty() const {
            return Length == 0;
        }

        constexpr T* data() const {
            return Data;
        }

        /// size - Get the array size.
        constexpr size_t size() const {
            return Length;
        }

        /// front - Get the first element.
        constexpr T& front() const {
            VERIFY(!empty(), "ArrayRef: attempted to access front() of empty list");
            return Data[0];
        }

        /// back - Get the last element.
        constexpr T& back() const {
            VERIFY(!empty(), "ArrayRef: attempted to access back() of empty list");
            return Data[Length - 1];
        }

        /// equals - Check for element-wise equality.
        constexpr bool equals(ArrayRef RHS) const {
            return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
        }

        /// slice(n, m) - Chop off the first N elements of the array, and keep M
        /// elements in the array.
        ArrayRef<T> slice(size_t N, size_t M) const {
            VERIFY(N + M <= size(),  "ArrayRef: invalid slice, N = " << N << "; M = " << M << "; size = " << size());
            return ArrayRef<T>(data() + N, M);
        }

        /// slice(n) - Chop off the first N elements of the array.
        constexpr ArrayRef<T> slice(size_t N) const {
            return slice(N, size() - N);
        }


        /// @}
        /// @name Operator Overloads
        /// @{
        constexpr const std::remove_const_t<T>& operator[](size_t Index) const {
            return Data[Index];
        }

        constexpr T& operator[](size_t Index) {
            return Data[Index];
        }

        /// Vector compatibility
        T& at(size_t Index) const {
            VERIFY(
                Index < Length,
                "ArrayRef: invalid index Index = " <<
                Index <<
                "; Length = " <<
                Length);
            return Data[Index];
        }

        /// Disallow accidental assignment from a temporary.
        ///
        /// The declaration here is extra complicated so that "arrayRef = {}"
        /// continues to select the move assignment operator.
        template <typename U>
        typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
        operator=(U&& Temporary) = delete;

        /// Disallow accidental assignment from a temporary.
        ///
        /// The declaration here is extra complicated so that "arrayRef = {}"
        /// continues to select the move assignment operator.
        template <typename U>
        typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
        operator=(std::initializer_list<U>) = delete;

        /// @}
    };

//    template <typename T>
//    std::ostream& operator<<(std::ostream & out, ArrayRef<T> list) {
//        int i = 0;
//        out << "[";
//        for(auto e : list) {
//            if (i++ > 0)
//                out << ", ";
//            out << e;
//        }
//        out << "]";
//        return out;
//    }


//    using IntList = ArrayRef<int64_t>;

}

template <class T>
using VecRef = Detail::ArrayRef<T>;


template <class T>
using ConstVecRef = Detail::ArrayRef<const T>;
