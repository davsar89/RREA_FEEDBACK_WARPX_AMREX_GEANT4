#pragma once
#include "AMReX_REAL.H"
#include <cstdio>
#include <cstdlib>
#include <string>
namespace amrex {
[[noreturn]] inline void Abort(char const* message){std::fprintf(stderr,"%s\n",message);std::abort();}
[[noreturn]] inline void Abort(std::string const& message){Abort(message.c_str());}
template<class... T> void ignore_unused(T const&...){}
}
