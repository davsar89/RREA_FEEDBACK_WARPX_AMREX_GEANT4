#ifndef RREA_KINETIC_SUPPORT_H
#define RREA_KINETIC_SUPPORT_H

// Storage and algorithms for the shared charged/photon collision kernels.
// Host-created vectors use the host heap; device-created vectors use the CUDA
// device heap. They MUST be packed on the device before host access. Numerical
// tables use immutable spans into managed arrays, not these owning vectors.
// Allocation failure is fatal, never a truncated collision history.
#include "rrea/RreaGpu.H"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <limits>
#include <type_traits>
#include <utility>
#include <initializer_list>

namespace rrea {

// A working type for function-local arithmetic only. CUDA has no extended
// floating-point arithmetic. Public signatures and packed/device-shared
// records MUST NOT depend on this alias. Helpers with wide arguments are
// templates so their declared signatures remain identical in both passes.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(RREA_KINETIC_TEST_BINARY64)
using RreaKineticWide = double;
#else
using RreaKineticWide = long double;
#endif

[[noreturn]] RREA_HOST_DEVICE inline void KineticFail(char const* message)
{
#if defined(__CUDA_ARCH__)
    printf("RREA kinetic kernel rejected state: %s\n", message);
    asm volatile("trap;");
    __builtin_unreachable();
#else
    std::fprintf(stderr, "RREA kinetic kernel rejected state: %s\n", message);
    std::abort();
#endif
}

template<class T> struct KineticSpan {
    T* ptr = nullptr;
    std::size_t length = 0;
    RREA_HOST_DEVICE T* begin() const noexcept { return ptr; }
    RREA_HOST_DEVICE T* end() const noexcept { return length ? ptr + length : ptr; }
    RREA_HOST_DEVICE T* data() const noexcept { return ptr; }
    RREA_HOST_DEVICE std::size_t size() const noexcept { return length; }
    RREA_HOST_DEVICE bool empty() const noexcept { return length == 0; }
    RREA_HOST_DEVICE T& operator[](std::size_t i) const noexcept { return ptr[i]; }
    RREA_HOST_DEVICE T& front() const noexcept { return ptr[0]; }
    RREA_HOST_DEVICE T& back() const noexcept { return ptr[length-1]; }
};

template<class T> class KineticVector {
    T* p_ = nullptr;
    std::size_t n_ = 0, cap_ = 0;
public:
    using value_type=T;
    using iterator=T*;
    using const_iterator=T const*;
    RREA_HOST_DEVICE KineticVector() noexcept = default;
    RREA_HOST_DEVICE explicit KineticVector(std::size_t n) { resize(n); }
    RREA_HOST_DEVICE KineticVector(std::size_t n,T const& value) { resize(n,value); }
    template<class It, std::enable_if_t<!std::is_integral_v<It>, int> = 0>
    RREA_HOST_DEVICE KineticVector(It first,It last) { for (; first!=last; ++first) push_back(*first); }
    KineticVector(std::initializer_list<T> values) : KineticVector(values.begin(),values.end()) {}
    RREA_HOST_DEVICE KineticVector(KineticVector const& v) : KineticVector(v.begin(),v.end()) {}
    RREA_HOST_DEVICE KineticVector(KineticVector&& v) noexcept { swap(v); }
    RREA_HOST_DEVICE ~KineticVector() { delete[] p_; }
    RREA_HOST_DEVICE KineticVector& operator=(KineticVector const& v) {
        if(this!=&v) { clear(); reserve(v.n_); for(auto const& x:v) push_back(x); } return *this;
    }
    RREA_HOST_DEVICE KineticVector& operator=(KineticVector&& v) noexcept {
        if(this!=&v) { KineticVector old; old.swap(*this); swap(v); } return *this;
    }
    RREA_HOST_DEVICE void swap(KineticVector& v) noexcept {
        auto* p=p_; p_=v.p_; v.p_=p;
        auto n=n_; n_=v.n_; v.n_=n;
        auto c=cap_; cap_=v.cap_; v.cap_=c;
    }
    RREA_HOST_DEVICE void reserve(std::size_t n) {
        if(n<=cap_) return;
        if(n>std::numeric_limits<std::size_t>::max()/sizeof(T)) KineticFail("vector size overflow");
        T* q=new T[n];
        if(!q) KineticFail("device heap exhausted; reduce kinetic batch size or increase heap bytes");
        for(std::size_t i=0;i<n_;++i) q[i]=static_cast<T&&>(p_[i]);
        delete[] p_; p_=q; cap_=n;
    }
    RREA_HOST_DEVICE std::size_t next_capacity() const {
        auto const limit=std::numeric_limits<std::size_t>::max()/sizeof(T);
        if(cap_>=limit)KineticFail("vector capacity overflow");
        auto const increment=cap_/2+1;
        return cap_==0 ? (limit<4 ? limit:4)
            : (increment>limit-cap_ ? limit:cap_+increment);
    }
    RREA_HOST_DEVICE void push_back(T const& x) {
        if(n_==cap_) {
            // Preserve append(v[0]) across a reallocation.
            T copy=x;reserve(next_capacity());p_[n_++]=static_cast<T&&>(copy);
        } else p_[n_++]=x;
    }
    RREA_HOST_DEVICE void push_back(T&& x) {
        if(n_==cap_) {
            T copy=static_cast<T&&>(x);reserve(next_capacity());p_[n_++]=static_cast<T&&>(copy);
        } else p_[n_++]=static_cast<T&&>(x);
    }
    RREA_HOST_DEVICE T& emplace_back() { push_back(T{}); return back(); }
    template<class... A> RREA_HOST_DEVICE T& emplace_back(A&&... a) {
        push_back(T(static_cast<A&&>(a)...)); return back();
    }
    RREA_HOST_DEVICE void resize(std::size_t n,T const& value=T{}) {
        if(n<n_) {
            for(auto i=n;i<n_;++i)p_[i]=T{};
        } else if(n>n_) {
            T saved=value; // value may alias an element that reserve relocates
            reserve(n);
            for(auto i=n_;i<n;++i)p_[i]=saved;
        }
        n_=n;
    }
    RREA_HOST_DEVICE void clear() { resize(0); }
    RREA_HOST_DEVICE T* erase(T* first,T* last) {
        if(first==last)return first;
        auto start=static_cast<std::size_t>(first-p_);
        auto removed=static_cast<std::size_t>(last-first);
        for(auto i=start;i+removed<n_;++i) p_[i]=static_cast<T&&>(p_[i+removed]);
        resize(n_-removed); return p_+start;
    }
    // Host adapters only: numerical kernels never accept an STL iterator.
    template<class It> void assign(It first,It last) {
        KineticVector copy(first,last);swap(copy);
    }
    template<class It> void insert(T* at,It first,It last) {
        if(at!=end())KineticFail("kinetic host insert only supports append");
        for(;first!=last;++first)push_back(*first);
    }
    RREA_HOST_DEVICE void pop_back() { if(!n_) KineticFail("empty vector pop"); p_[--n_]=T{}; }
    RREA_HOST_DEVICE T* begin() noexcept { return p_; }
    RREA_HOST_DEVICE T* end() noexcept { return n_ ? p_+n_ : p_; }
    RREA_HOST_DEVICE T const* begin() const noexcept { return p_; }
    RREA_HOST_DEVICE T const* end() const noexcept { return n_ ? p_+n_ : p_; }
    RREA_HOST_DEVICE T* data() noexcept { return p_; }
    RREA_HOST_DEVICE T const* data() const noexcept { return p_; }
    RREA_HOST_DEVICE std::size_t size() const noexcept { return n_; }
    RREA_HOST_DEVICE bool empty() const noexcept { return n_==0; }
    RREA_HOST_DEVICE T& operator[](std::size_t i) noexcept { return p_[i]; }
    RREA_HOST_DEVICE T const& operator[](std::size_t i) const noexcept { return p_[i]; }
    RREA_HOST_DEVICE T& front() noexcept { return p_[0]; }
    RREA_HOST_DEVICE T const& front() const noexcept { return p_[0]; }
    RREA_HOST_DEVICE T& back() noexcept { return p_[n_-1]; }
    RREA_HOST_DEVICE T const& back() const noexcept { return p_[n_-1]; }
};

namespace kinetic {
template<class It> RREA_HOST_DEVICE It adjacent_find(It first,It last) {
    if(first==last)return last;
    for(It next=first+1;next!=last;++next,++first)if(*first==*next)return first;
    return last;
}

RREA_HOST_DEVICE inline long long abs_id(long long v){return v<0?-v:v;}

template<class... T> RREA_HOST_DEVICE void ignore_unused(T const&...) {}

template<class T> RREA_HOST_DEVICE T clamp(T x,T lo,T hi){return x<lo?lo:(x>hi?hi:x);}

template<class T> RREA_HOST_DEVICE constexpr T min(T a,T b) { return b<a ? b:a; }
template<class T> RREA_HOST_DEVICE constexpr T max(T a,T b) { return a<b ? b:a; }
template<class T> RREA_HOST_DEVICE T max(std::initializer_list<T> xs) {
    auto it=xs.begin();T value=*it++;for(;it!=xs.end();++it)if(value<*it)value=*it;return value;
}
template<class It,class Equal> RREA_HOST_DEVICE It unique(It first,It last,Equal equal) {
    if(first==last)return last;
    It out=first;for(It it=first+1;it!=last;++it)if(!equal(*out,*it))*++out=*it;
    return out+1;
}

template<class T> RREA_HOST_DEVICE void swap(T& a,T& b) { T tmp=static_cast<T&&>(a); a=static_cast<T&&>(b); b=static_cast<T&&>(tmp); }
struct Less { template<class A,class B> RREA_HOST_DEVICE bool operator()(A const& a,B const& b) const { return a<b; } };
template<class It,class T,class LessT=Less>
RREA_HOST_DEVICE It lower_bound(It first,It last,T const& v,LessT less={}) {
    auto count=first==last ? 0 : last-first;
    while(count>0) { auto step=count/2; auto mid=first+step;
        if(less(*mid,v)) { first=mid+1; count-=step+1; } else count=step;
    } return first;
}
template<class It,class T,class LessT=Less>
RREA_HOST_DEVICE It upper_bound(It first,It last,T const& v,LessT less={}) {
    auto count=first==last ? 0 : last-first;
    while(count>0) { auto step=count/2; auto mid=first+step;
        if(!less(v,*mid)) { first=mid+1; count-=step+1; } else count=step;
    } return first;
}
template<class It> RREA_HOST_DEVICE auto distance(It a,It b) { return a==b ? 0 : b-a; }
// In-place heapsort, bounded stack and no allocation. Stable identifiers are
// included in every population comparator; tie order is not an RNG input.
template<class It,class LessT=Less>
RREA_HOST_DEVICE void sort(It first,It last,LessT less={}) {
    auto const n=first==last ? 0 : last-first;
    auto sift=[&](std::ptrdiff_t root,std::ptrdiff_t end) {
        while(2*root+1<end) { auto c=2*root+1;
            if(c+1<end && less(first[c],first[c+1])) ++c;
            if(!less(first[root],first[c])) break;
            kinetic::swap(first[root],first[c]); root=c;
        }
    };
    for(auto start=n/2;start>0;--start) sift(start-1,n);
    for(auto end=n;end>1;--end) { kinetic::swap(first[0],first[end-1]); sift(0,end-1); }
}
template<class It> RREA_HOST_DEVICE It unique(It first,It last) {
    if(first==last) return last;
    auto out=first;
    for(auto p=first+1;p!=last;++p) if(!(*out==*p)) *++out=*p;
    return out+1;
}
template<class It,class Pred> RREA_HOST_DEVICE std::size_t count_if(It first,It last,Pred pred) {
    std::size_t n=0; for(;first!=last;++first) if(pred(*first)) ++n; return n;
}
template<class It,class LessT=Less> RREA_HOST_DEVICE It max_element(It first,It last,LessT less={}) {
    if(first==last) return last;
    auto largest=first;
    for(auto it=first+1;it!=last;++it) if(less(*largest,*it)) largest=it;
    return largest;
}
} // kinetic
} // rrea
#endif
