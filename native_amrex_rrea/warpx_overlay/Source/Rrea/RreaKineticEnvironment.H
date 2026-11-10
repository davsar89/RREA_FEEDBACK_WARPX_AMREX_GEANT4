#ifndef RREA_KINETIC_ENVIRONMENT_H
#define RREA_KINETIC_ENVIRONMENT_H
#include "rrea/RreaAltitudeProfile.H"
#include "rrea/RreaKineticFieldView.H"
#include "rrea/RreaElectronCreation.H"
#include <cstdint>
namespace rrea::warpx {
struct RreaFieldSample {
    amrex::Real er_v_per_m = amrex::Real(0.0);
    amrex::Real ez_v_per_m = amrex::Real(0.0);
    amrex::Real btheta_t = amrex::Real(0.0);
};
// Air density along z relative to the transport tables' reference density:
// the altitude profile over that reference where a profile is loaded, the
// deck's scalar ratio (already table-relative) otherwise.
struct TransportDensityView {
    rrea::RreaAltitudeProfileView profile;
    amrex::Real scalar_ratio=1, reference_density=1, altitude_at_z0=0;
    bool profile_enabled=false;
    RREA_HOST_DEVICE amrex::Real RatioAtZ(amrex::Real z) const {
        if(!profile_enabled)return scalar_ratio;
        if(!(reference_density>0))rrea::KineticFail("RREA transport reference density invalid");
        return profile.Interpolate(altitude_at_z0+z)/reference_density;
    }
};
struct KineticEnvironment {
    rrea::RreaKineticFieldView field;
    TransportDensityView density;
    rrea::KineticSpan<amrex::Real const> planes;
    std::uint64_t seed=0;
    amrex::Real low_cutoff=0, moller_cutoff=0, photon_cutoff=0;
    bool planes_enabled=false;
    bool electron_creation_enabled=false;
    rrea::KineticResource allocation{};
    RREA_HOST_DEVICE std::uint64_t RngSeed() const {return seed;}
    RREA_HOST_DEVICE amrex::Real LowEnergyCutoffEv() const {return low_cutoff;}
    RREA_HOST_DEVICE amrex::Real HardMollerSecondaryThresholdEv() const {return moller_cutoff;}
    RREA_HOST_DEVICE amrex::Real PhotonCutoffEv() const {return photon_cutoff;}
    RREA_HOST_DEVICE bool CdPlaneDiagnosticsEnabled() const {return planes_enabled;}
    RREA_HOST_DEVICE auto const& CdPlaneZ() const {return planes;}
    RREA_HOST_DEVICE amrex::Real TransportDensityRatioAtZ(amrex::Real z) const {
        return density.RatioAtZ(z);
    }
    RREA_HOST_DEVICE RreaFieldSample GatherFieldRZ(double r,double z) const {
        if(!field.field.er || !field.field.ez || !field.field.btheta)
            rrea::KineticFail("RREA kinetic field gather has no initialized field");
        double er,ez,bt;
        field.Gather(r,z,er,ez,bt);
        return {er,ez,bt};
    }
};
}
#endif
