// Only linked by the standalone scalar test script. Native builds use the real
// ParmParse-backed debug configuration from RreaDebugOptions.cpp.
#include "AMReX_REAL.H"
#include "rrea/RreaDebugOptions.H"
namespace rrea {
RreaDebugTransportSensitivityOptions const& DebugTransportSensitivityOptions(){
    static RreaDebugTransportSensitivityOptions const defaults{};
    return defaults;
}
}
