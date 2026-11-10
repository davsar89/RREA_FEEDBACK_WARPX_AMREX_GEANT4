#ifndef RREA_KINETIC_SUPPORT_H
#define RREA_KINETIC_SUPPORT_H

// Storage and algorithms for the shared charged/photon collision kernels.
// Host-created vectors use the host heap; device-created vectors use the CUDA
// device heap. They MUST be packed on the device before host access. Numerical
// tables use immutable spans into managed arrays, not these owning vectors.
// Allocation failure is fatal, never a truncated collision history.
#include "rrea/RreaGpu.H"
#include <cstddef>
#include <cstdint>
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

// Explicit per-history resource. No global current allocator and no TLS.
// A null base selects the original heap, optionally counting the monotonic
// capacity that a pool would need. All pooled objects die before reuse.
struct KineticArena {
    unsigned char* base = nullptr;
    std::size_t limit = 0, used = 0, allocations = 0;
    bool audit = false;
    bool allow_spill = true;
    std::size_t pool_used = 0, spills = 0, spill_bytes = 0;
    static constexpr std::size_t heap_offset =
        std::numeric_limits<std::size_t>::max();

    RREA_HOST_DEVICE std::size_t claim(
        std::size_t bytes, std::size_t alignment)
    {
        if (!base && !audit) return heap_offset;
        auto const max = heap_offset;
        if (!bytes || !alignment || alignment > 256
            || (alignment & (alignment-1)) || used > max-(alignment-1))
            KineticFail("invalid kinetic arena allocation");

        auto const demand_offset = (used+alignment-1) & ~(alignment-1);
        if (bytes > max-demand_offset || allocations == max)
            KineticFail("kinetic arena demand overflow");

        // Hypothetical all-pool demand, including spilled allocations.
        used = demand_offset+bytes;
        if (audit) ++allocations;
        if (!base) return heap_offset;

        if (reinterpret_cast<std::uintptr_t>(base) & (alignment-1))
            KineticFail("misaligned kinetic arena storage");
        if (pool_used > limit || pool_used > max-(alignment-1))
            KineticFail("invalid kinetic pool cursor");

        auto const offset = (pool_used+alignment-1) & ~(alignment-1);
        if (offset <= limit && bytes <= limit-offset) {
            pool_used = offset+bytes;
            return offset;
        }

        if (!allow_spill) {
            printf("RREA kinetic arena exhausted: request %llu, cursor %llu, capacity %llu\n",
                   static_cast<unsigned long long>(bytes),
                   static_cast<unsigned long long>(pool_used),
                   static_cast<unsigned long long>(limit));
            KineticFail("kinetic arena exhausted; no history may be truncated");
        }

        if (spills == max || bytes > max-spill_bytes)
            KineticFail("kinetic spill counter overflow");
        ++spills;
        spill_bytes += bytes;

        // Preserve remaining space for subsequent smaller requests.
        return heap_offset;
    }
};
struct KineticResource { KineticArena* arena = nullptr; };

template<class T> class KineticVector {
    T* p_ = nullptr;
    std::size_t n_ = 0, cap_ = 0;
    KineticArena* arena_ = nullptr;
    bool pooled_ = false;
    RREA_HOST_DEVICE void release() noexcept {
        if (!p_) return;
        if (pooled_) {
            // new T[capacity] constructed every slot, not just size() slots.
            for (auto i=cap_; i>0; --i) p_[i-1].~T();
        } else {
            delete[] p_;
        }
    }
public:
    using value_type=T;
    using iterator=T*;
    using const_iterator=T const*;
    RREA_HOST_DEVICE KineticVector() noexcept = default;
    RREA_HOST_DEVICE explicit KineticVector(KineticResource r) noexcept
        : arena_(r.arena) {}
    RREA_HOST_DEVICE void bind(KineticResource r) {
        if (p_ || n_ || cap_)
            KineticFail("cannot rebind allocated kinetic vector");
        arena_=r.arena;
    }
    RREA_HOST_DEVICE explicit KineticVector(std::size_t n) { resize(n); }
    RREA_HOST_DEVICE KineticVector(std::size_t n,T const& value) { resize(n,value); }
    template<class It, std::enable_if_t<!std::is_integral_v<It>, int> = 0>
    RREA_HOST_DEVICE KineticVector(It first,It last) {
        // Support spans and vector copies have a known size. Avoid repeated
        // device-heap allocation/move/destruction while importing the range.
        if constexpr (std::is_pointer_v<It>) {
            if(first!=last) reserve(static_cast<std::size_t>(last-first));
        }
        for (; first!=last; ++first) push_back(*first);
    }
    KineticVector(std::initializer_list<T> values) : KineticVector(values.begin(),values.end()) {}
    RREA_HOST_DEVICE KineticVector(KineticVector const& v)
        : arena_(v.arena_) {
        reserve(v.n_);
        for (auto const& x:v) push_back(x);
    }
    RREA_HOST_DEVICE KineticVector(KineticVector&& v) noexcept { swap(v); }
    RREA_HOST_DEVICE ~KineticVector() { release(); }
    RREA_HOST_DEVICE KineticVector& operator=(KineticVector const& v) {
        if(this!=&v) {
            if (arena_!=v.arena_) {
                KineticVector copy(v); swap(copy);
            } else {
                clear(); reserve(v.n_);
                for(auto const& x:v) push_back(x);
            }
        }
        return *this;
    }
    RREA_HOST_DEVICE KineticVector& operator=(KineticVector&& v) noexcept {
        if(this!=&v) { KineticVector old; old.swap(*this); swap(v); } return *this;
    }
    RREA_HOST_DEVICE void swap(KineticVector& v) noexcept {
        auto* p=p_; p_=v.p_; v.p_=p;
        auto n=n_; n_=v.n_; v.n_=n;
        auto c=cap_; cap_=v.cap_; v.cap_=c;
        auto a=arena_; arena_=v.arena_; v.arena_=a;
        auto b=pooled_; pooled_=v.pooled_; v.pooled_=b;
    }
    RREA_HOST_DEVICE void reserve(std::size_t n) {
        if(n<=cap_) return;
        if(n>std::numeric_limits<std::size_t>::max()/sizeof(T)) KineticFail("vector size overflow");
        T* q=nullptr;
        std::size_t offset=KineticArena::heap_offset;
        if (arena_) offset=arena_->claim(n*sizeof(T),alignof(T));
        bool const from_pool=offset!=KineticArena::heap_offset;
        if (from_pool) {
            q=reinterpret_cast<T*>(arena_->base+offset);
            for (std::size_t i=0; i<n; ++i)
                ::new (static_cast<void*>(q+i)) T;
        } else {
            q=new T[n];
        }
        if(!q) KineticFail("device heap exhausted; reduce kinetic batch size or increase heap bytes");
        for(std::size_t i=0;i<n_;++i) q[i]=static_cast<T&&>(p_[i]);
        release(); p_=q; cap_=n; pooled_=from_pool;
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

// splitmix64 (Steele, Lea & Flood 2014): the one 64-bit mixer behind the
// counter RNG, the rejection-sampler stream and the CIC support keys.
RREA_HOST_DEVICE inline std::uint64_t splitmix64(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

namespace kinetic {
RREA_HOST_DEVICE inline long long abs_id(long long v){return v<0?-v:v;}
// The one particle key: folding the birth-rank cpu() into the id keeps per-rank
// id reuse from colliding RNG streams or staged parent removals.
RREA_HOST_DEVICE inline std::uint64_t stable_particle_key(long long id, int cpu)
{
    return static_cast<std::uint64_t>(abs_id(id)) ^ (static_cast<std::uint64_t>(cpu) << 40);
}

template<class... T> RREA_HOST_DEVICE void ignore_unused(T const&...) {}

template<class T> RREA_HOST_DEVICE T clamp(T x,T lo,T hi){return x<lo?lo:(x>hi?hi:x);}

template<class T> RREA_HOST_DEVICE constexpr T min(T a,T b) { return b<a ? b:a; }
template<class T> RREA_HOST_DEVICE constexpr T max(T a,T b) { return a<b ? b:a; }
template<class T> RREA_HOST_DEVICE T max(std::initializer_list<T> xs) {
    auto it=xs.begin();T value=*it++;for(;it!=xs.end();++it)if(value<*it)value=*it;return value;
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
