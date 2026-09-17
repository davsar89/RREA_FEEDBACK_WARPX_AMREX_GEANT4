! parma_electron_query: thin batch evaluator for the PARMA cosmic-ray model.
!
! Physics owner: scripts/rrea_parma_source.py (column-normalized RREA seed
! source).  This program only evaluates the upstream PARMA parametrization on
! grids supplied by Python via stdin; it owns no quadrature, normalization,
! or RREA geometry.  Link against the vendored rrea_parma_upstream/
! subroutines.f90 (from github.com/davsar89/COSMIC_RAY_FLUXES, tracked in
! this repository) and run with that directory as the working directory
! (the model opens its data as 'input/...').
!
! Model citations:
!   Sato (2015) PLoS ONE 10, e0144679 (PARMA/EXPACS spectra, getSpec)
!   Sato (2016) PLoS ONE 11, e0160390 (PARMA4 angular model, getSpecAngFinal;
!     mu = cos(theta) = 1 is vertically DOWNWARD; the electron and positron
!     angular distributions share one parametrization, particle class 5)
! The PARMA/EXPACS license permits research use with citation; commercial use
! requires prior agreement with JAEA.
!
! Fixed model state (deliberate; do not read the clock):
!   solar reference 2019-05-27 -- the most recent valid daily W-index in the
!   bundled tables, and the state behind the recorded upstream anchor fluxes;
!   local geometry parameter g = 0.15; US Standard 1976 atmosphere;
!   spectral IDs 31 electron and 32 positron (getSpec), angular class 5
!   'elepos' (one shared e-/e+ law).
!
! stdin (list-directed):  lat lon / nz / alt_km(nz) / ne / e_mev(ne) /
!                         nea / ea_mev(nea) / nmu / mu(nmu)
! stdout: PARMA_QUERY_V2 header, SOLAR/RIGIDITY_GV lines, then DEPTH_G_CM2,
!         SPEC31 and SPEC32 (z-major) and ANG (z, then energy, then mu)
!         value blocks, one ES25.17E3 value per line.
program parma_electron_query
   use, intrinsic :: iso_fortran_env, only: real64, error_unit, output_unit
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   implicit none

   interface
      function getHP(year, month, day, status_code) result(value)
         import real64
         integer, intent(in) :: year, month, day
         integer, intent(out) :: status_code
         real(real64) :: value
      end function getHP

      function getr(latitude, longitude) result(value)
         import real64
         real(real64), intent(in) :: latitude, longitude
         real(real64) :: value
      end function getr

      function getd_model(altitude, latitude, model, status_code) result(value)
         import real64
         real(real64), intent(in) :: altitude, latitude
         integer, intent(in) :: model
         integer, intent(out) :: status_code
         real(real64) :: value
      end function getd_model

      function getSpec(particle_id, solar_index, cutoff_rigidity, atmospheric_depth, energy, g) result(value)
         import real64
         integer, intent(in) :: particle_id
         real(real64), intent(in) :: solar_index, cutoff_rigidity, atmospheric_depth, energy, g
         real(real64) :: value
      end function getSpec

      function getSpecAngFinal(particle_class, solar_index, cutoff_rigidity, atmospheric_depth, energy, g, mu) result(value)
         import real64
         integer, intent(in) :: particle_class
         real(real64), intent(in) :: solar_index, cutoff_rigidity, atmospheric_depth, energy, g, mu
         real(real64) :: value
      end function getSpecAngFinal
   end interface

   integer, parameter :: solar_year = 2019, solar_month = 5, solar_day = 27
   integer, parameter :: electron_particle_id = 31
   integer, parameter :: positron_particle_id = 32
   integer, parameter :: angular_particle_class = 5
   integer, parameter :: atmosphere_us76 = 0
   real(real64), parameter :: geometry = 0.15_real64

   integer :: solar_status, depth_status, nz, ne, nea, nmu, iz, ie, imu, isp
   integer :: species_id
   real(real64) :: lat, lon, solar_w, rigidity, value
   real(real64), allocatable :: alt_km(:), energy_mev(:), ang_energy_mev(:), mu(:), depth(:)

   read (*, *) lat, lon
   read (*, *) nz
   if (nz < 2) call fail('need at least two altitude nodes')
   allocate (alt_km(nz), depth(nz))
   read (*, *) alt_km
   read (*, *) ne
   if (ne < 2) call fail('need at least two spectral energy nodes')
   allocate (energy_mev(ne))
   read (*, *) energy_mev
   read (*, *) nea
   if (nea < 2) call fail('need at least two angular energy nodes')
   allocate (ang_energy_mev(nea))
   read (*, *) ang_energy_mev
   read (*, *) nmu
   if (nmu < 2) call fail('need at least two mu nodes')
   allocate (mu(nmu))
   read (*, *) mu

   solar_w = getHP(solar_year, solar_month, solar_day, solar_status)
   if (solar_status < 1 .or. solar_status > 3) then
      call fail('solar-activity lookup failed for the frozen 2019-05-27 reference')
   end if
   rigidity = getr(lat, lon)
   if (.not. ieee_is_finite(rigidity) .or. rigidity < 0.0_real64) then
      call fail('cut-off rigidity calculation failed for the requested location')
   end if
   do iz = 1, nz
      depth(iz) = getd_model(alt_km(iz), lat, atmosphere_us76, depth_status)
      if (depth_status /= 0 .or. .not. ieee_is_finite(depth(iz)) .or. depth(iz) <= 0.0_real64) then
         call fail('atmospheric-depth calculation failed for a requested altitude')
      end if
   end do

   write (output_unit, '(A)') 'PARMA_QUERY_V2'
   write (output_unit, '(A,I0,1X,I0,1X,I0,1X,ES25.17E3,1X,I0)') &
      'SOLAR ', solar_year, solar_month, solar_day, solar_w, solar_status
   write (output_unit, '(A,ES25.17E3)') 'RIGIDITY_GV ', rigidity
   write (output_unit, '(A,I0)') 'DEPTH_G_CM2 ', nz
   do iz = 1, nz
      write (output_unit, '(ES25.17E3)') depth(iz)
   end do
   do isp = 1, 2
      if (isp == 1) then
         species_id = electron_particle_id
      else
         species_id = positron_particle_id
      end if
      write (output_unit, '(A,I0,1X,I0,1X,I0)') 'SPEC', species_id, nz, ne
      do iz = 1, nz
         do ie = 1, ne
            value = getSpec(species_id, solar_w, rigidity, depth(iz), energy_mev(ie), geometry)
            if (.not. ieee_is_finite(value) .or. value < 0.0_real64) then
               call fail('the spectral model returned an invalid differential flux')
            end if
            write (output_unit, '(ES25.17E3)') value
         end do
      end do
   end do
   ! mu innermost keeps getSpecAng's (ip,s,r,d,e,g) parameter cache hot.
   write (output_unit, '(A,I0,1X,I0,1X,I0)') 'ANG ', nz, nea, nmu
   do iz = 1, nz
      do ie = 1, nea
         do imu = 1, nmu
            value = getSpecAngFinal(angular_particle_class, solar_w, rigidity, &
                                    depth(iz), ang_energy_mev(ie), geometry, mu(imu))
            if (.not. ieee_is_finite(value) .or. value < 0.0_real64) then
               call fail('the angular model returned an invalid value')
            end if
            write (output_unit, '(ES25.17E3)') value
         end do
      end do
   end do

contains

   subroutine fail(message)
      character(len=*), intent(in) :: message
      write (error_unit, '(A)') 'ERROR: '//trim(message)
      error stop 1
   end subroutine fail

end program parma_electron_query
