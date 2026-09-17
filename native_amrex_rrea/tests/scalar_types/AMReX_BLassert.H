#pragma once
#include "AMReX.H"
#define AMREX_ALWAYS_ASSERT(x) do {if(!(x))amrex::Abort("scalar test assertion: " #x);} while(false)
#define AMREX_ALWAYS_ASSERT_WITH_MESSAGE(x,m) do {if(!(x))amrex::Abort(m);} while(false)
#define BL_ASSERT(x) AMREX_ALWAYS_ASSERT(x)
