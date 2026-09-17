#pragma once
#ifndef RREA_SCALAR_TEST_ADAPTERS
# error "Scalar test aliases must never shadow the real AMReX headers in a production build"
#endif
// Test-only scalar names, not a mesh/particle runtime or a CUDA emulator.
namespace amrex { using Real=double; using ParticleReal=double; using Long=long long; }
