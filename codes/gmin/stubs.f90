! Glue around GMIN mymylbfgs. The modules below satisfy the USE lists in
! source/mymylbfgs.F90. Flags that would leave the Cartesian LBFGS path
! stay false. POTENTIAL is the harness callback.
!
! GMIN is free software; you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the Free
! Software Foundation; either version 2 of the License, or (at your
! option) any later version.

module prec
   integer, parameter :: real64 = selected_real_kind(15, 307)
end module prec

module commons
   use prec, only: real64
   implicit none
   integer :: which_pot = 0
   integer :: myunit = 6
   integer :: ns = 0
   integer :: gthommet = 0
   integer :: qiter = 0
   integer :: ncore = 0
   integer :: prtfrq = 1
   integer, allocatable :: nq(:)
   integer, allocatable :: dumpxyzunit(:)
   integer, allocatable :: dumpcartxyzunit(:)
   real(real64) :: rmsshift(6) = 0.0d0
   real(real64) :: coldfusionlimit = -1.0d100
   real(real64) :: dguess = 0.1d0
   real(real64) :: energycamshift = 0.0d0
   real(real64) :: maxbfgs = 0.4d0
   real(real64) :: maxefall = -1.0d300
   real(real64) :: maxerise = 1.0d-10
   real(real64) :: rms = 1.0d0
   real(real64) :: napshiftenergy = 0.0d0
   real(real64) :: napshiftrms = 0.0d0
   real(real64) :: ffenergy = 0.0d0
   real(real64) :: ffrms = 0.0d0
   real(real64) :: dc_energy = 0.0d0
   real(real64) :: dc_rms = 0.0d0
   real(real64) :: ffenergy_ns = 0.0d0
   real(real64) :: ffrms_ns = 0.0d0
   real(real64) :: ffenergy_dc = 0.0d0
   real(real64) :: ffrms_dc = 0.0d0
   real(real64) :: ffenergy_nap_dc = 0.0d0
   real(real64) :: ffrms_nap_dc = 0.0d0
   real(real64) :: guidesymconv = -1.0d0
   real(real64), allocatable :: celldata(:)
   real(real64), allocatable :: atmass(:)
   character(len=2), allocatable :: zsym(:)
   logical :: boxderivt = .false.
   logical :: calcqt = .false.
   logical :: cent = .false.
   logical :: chemshift = .false.
   logical :: chemshift2 = .false.
   logical :: coldfusion = .false.
   logical :: dumpt = .false.
   logical :: fixdiheflag = .false.
   logical :: fiximage = .false.
   logical :: gcbht = .true.
   logical :: intmint = .false.
   logical :: ortho = .false.
   logical :: projiht = .false.
   logical :: projit = .false.
   logical :: lmperr = .false.
   logical :: pycoldfusion = .false.
   logical :: usenapshift = .false.
   logical :: usedipolarcoupling = .false.
   logical :: usefract = .false.
   logical :: periodic = .false.
   logical :: guidesymt = .false.
   logical :: lateralpressuret = .false.
   integer, parameter :: pamber = 1
   integer, parameter :: pamber12 = 2
   integer, parameter :: pcharmm = 3
   integer, parameter :: parno = 4
   integer, parameter :: pdbp = 5
   integer, parameter :: pdbptd = 6
   integer, parameter :: pdmacrys = 7
   integer, parameter :: pdmblpy = 8
   integer, parameter :: pgromacs = 9
   integer, parameter :: pgthomson = 10
   integer, parameter :: pthomson = 11
   integer, parameter :: pjm = 12
   integer, parameter :: pmbpol = 13
   integer, parameter :: pmsstock = 14
   integer, parameter :: pmstbin = 15
   integer, parameter :: pmultpaha = 16
   integer, parameter :: pnpah = 17
   integer, parameter :: ppacheco = 18
   integer, parameter :: ppahw99 = 19
   integer, parameter :: ppy = 20
   integer, parameter :: ppybinary = 21
   integer, parameter :: ppygperiodic = 22
   integer, parameter :: prgcl2 = 23
   integer, parameter :: psilane = 24
   integer, parameter :: ptdhd = 25
   integer, parameter :: ptip = 26
   integer, parameter :: ptosi = 27
   integer, parameter :: pttm3 = 28
   integer, parameter :: pwelch = 29
   integer, parameter :: pxtb = 30
   integer, parameter :: pxtbapi = 31
   integer, parameter :: pdftbp = 32
   integer, parameter :: plammps = 33
   integer, parameter :: pioniccolloid = 34
   integer, parameter :: pase = 35
   integer, parameter :: ppippot = 36
   integer, parameter :: pamber20 = 37
   integer, parameter :: ppyplate = 38

contains

   integer function get_nbox()
      get_nbox = 0
   end function get_nbox

   function get_box_index() result(index)
      integer :: index(6)
      index = 0
   end function get_box_index

end module commons

module genrigid
   use prec, only: real64
   implicit none
   integer :: natoms = 0
   integer :: nrigidbody = 0
   integer :: degfreedoms = 0
   logical :: atomrigidcoordt = .true.
   logical :: rigidinit = .false.
   real(real64) :: box_params(6) = 0.0d0
   real(real64) :: box_paramsgrad(6) = 0.0d0

contains

   subroutine transformctorigid(x, xr)
      real(real64), intent(in) :: x(:)
      real(real64), intent(out) :: xr(:)
      xr = x
   end subroutine transformctorigid

   subroutine transformrigidtoc(a, b, x, xr)
      integer, intent(in) :: a, b
      real(real64), intent(inout) :: x(:)
      real(real64), intent(in) :: xr(:)
   end subroutine transformrigidtoc

end module genrigid

module modcharmm
   implicit none
   logical :: gcharmmfail = .false.
   logical :: acesolv = .false.
   integer :: nchencalls = 0
   integer :: aceupstep = 0
end module modcharmm

module porfuncs
   implicit none
   logical :: debug = .false.
end module porfuncs

module cartdist
   implicit none
end module cartdist

module modamber9
   use prec, only: real64
   implicit none
   logical :: steeredmint = .false.
   logical :: localsteeredmint = .false.
   real(real64) :: smink = 0.0d0
   real(real64) :: sminkinc = 0.0d0
   real(real64) :: sminkcurrent = 0.0d0
end module modamber9

module gmin_helpers
   implicit none
   logical :: guidechanget = .false.
   logical :: guidet = .false.
   logical :: ftest = .false.
end module gmin_helpers

module mc_helpers
   use prec, only: real64
   implicit none
   real(real64) :: potel = 0.0d0
   real(real64) :: qstart = 0.0d0
   logical :: evap = .false.
   logical :: evapreject = .false.
end module mc_helpers

module finalio_mod
   implicit none
contains
   subroutine tipio(a, b, c, d, e, f, rb)
      use prec, only: real64
      real(real64), intent(in) :: a, b, c, d, e, f
      real(real64), intent(out) :: rb(:)
      rb = 0.0d0
   end subroutine tipio
end module finalio_mod

module interfaces_export
   implicit none
contains
   subroutine dftbp_init()
   end subroutine dftbp_init
end module interfaces_export

module q4module
   implicit none
contains
   subroutine orderq4(natoms, x, qstart)
      use prec, only: real64
      integer, intent(in) :: natoms
      real(real64), intent(in) :: x(:)
      real(real64), intent(in) :: qstart
   end subroutine orderq4
end module q4module

module thomson_mod
   implicit none
contains
   subroutine thomsonctoang(a, b, n, unit)
      use prec, only: real64
      real(real64), intent(in) :: a(:)
      real(real64), intent(out) :: b(:)
      integer, intent(in) :: n, unit
      b = 0.0d0
   end subroutine thomsonctoang

   subroutine thomsonangtoc(a, n)
      use prec, only: real64
      real(real64), intent(inout) :: a(:)
      integer, intent(in) :: n
   end subroutine thomsonangtoc
end module thomson_mod

module box_derivatives
   implicit none
contains
   subroutine frac2cart(a, b)
      use prec, only: real64
      real(real64), intent(out) :: a(:)
      real(real64), intent(in) :: b(:)
      a = b
   end subroutine frac2cart
end module box_derivatives

module gmin_eval
   use iso_c_binding
   use prec, only: real64
   implicit none
   real(real64) :: harness_eps = 0.0d0

   abstract interface
      integer(c_int) function eval_iface(n, x, energy, grad, hold) bind(c)
         import :: c_int, c_double
         integer(c_int), value :: n
         real(c_double), intent(in) :: x(n)
         real(c_double), intent(out) :: energy
         real(c_double), intent(out) :: grad(n)
         integer(c_int), intent(out) :: hold
      end function eval_iface
   end interface

   procedure(eval_iface), pointer :: eval_ptr => null()
end module gmin_eval

subroutine potential(x, grad, ereal, gradt, sect)
   use prec, only: real64
   use genrigid, only: natoms
   use commons, only: rms
   use gmin_helpers, only: ftest
   use gmin_eval, only: eval_ptr, harness_eps
   implicit none
   real(real64), intent(in) :: x(3*natoms)
   real(real64), intent(out) :: grad(3*natoms)
   real(real64), intent(out) :: ereal
   logical, intent(in) :: gradt
   logical, intent(in) :: sect
   integer :: n, hold, status
   real(real64) :: dummy

   n = 3*natoms
   hold = 0
   if (.not. associated(eval_ptr)) then
      ftest = .true.
      ereal = 0.0d0
      grad = 0.0d0
      rms = 1.0d0
      return
   end if
   status = eval_ptr(n, x(1:n), ereal, grad(1:n), hold)
   if (status /= 0) then
      ftest = .true.
      ereal = 0.0d0
      grad = 0.0d0
      rms = 1.0d0
      return
   end if
   dummy = sum(grad(1:n)**2)
   rms = max(sqrt(dummy / real(n, real64)), 1.0d-100)
   ! hold 1: the harness stop is not met, so a small RMS must not return.
   ! hold 2: the harness stop is met. The gradient is unchanged. RMS is
   ! brought under EPS so the termination test returns on this point.
   if (hold == 1 .and. rms <= harness_eps) then
      rms = harness_eps * (1.0d0 + 1.0d-8)
   else if (hold == 2 .and. harness_eps > 0.0d0) then
      if (rms > harness_eps * (1.0d0 - 1.0d-12)) rms = harness_eps * (1.0d0 - 1.0d-12)
   end if
end subroutine potential

subroutine updatenbonds(x)
   use prec, only: real64
   real(real64) :: x(*)
end subroutine updatenbonds

subroutine getkd(kd)
   integer :: kd
   kd = 0
end subroutine getkd

subroutine getnnz(nnz)
   integer :: nnz
   nnz = 0
end subroutine getnnz

subroutine transform(x, g, xi, gi, n, ncart, nnz, nocoor, kd)
   use prec, only: real64
   real(real64) :: x(*), g(*), xi(*), gi(*)
   integer :: n, ncart, nnz, kd
   logical :: nocoor
end subroutine transform

subroutine dmacrys_reduce_cell_rigid(x, changed)
   use prec, only: real64
   real(real64) :: x(*)
   logical :: changed
   changed = .false.
end subroutine dmacrys_reduce_cell_rigid

subroutine proji(x, n)
   use prec, only: real64
   real(real64) :: x(*)
   integer :: n
end subroutine proji

subroutine projih(x, n)
   use prec, only: real64
   real(real64) :: x(*)
   integer :: n
end subroutine projih

subroutine gthomsonangtoc(a, b, n)
   use prec, only: real64
   real(real64) :: a(*), b(*)
   integer :: n
end subroutine gthomsonangtoc

subroutine gthomsonctoang(a, b, n)
   use prec, only: real64
   real(real64) :: a(*), b(*)
   integer :: n
end subroutine gthomsonctoang

subroutine transbackdelta(dq, dc, lc, n, ncart, nnz, kd, failed, flag, eps)
   use prec, only: real64
   real(real64) :: dq(*), dc(*), lc(*), eps
   integer :: n, ncart, nnz, kd
   logical :: failed, flag
   failed = .true.
end subroutine transbackdelta

subroutine centre2(x)
   use prec, only: real64
   real(real64) :: x(*)
end subroutine centre2

subroutine dumpprint(x, np, iter)
   use prec, only: real64
   real(real64) :: x(*)
   integer :: np, iter
end subroutine dumpprint

! The driver is compiled after lbfgs_mod. This marker keeps the callback
! registration next to the potential. The run entry point is driver.f90.
subroutine gmin_set_eval(ptr) bind(c, name="gmin_set_eval")
   use iso_c_binding
   use gmin_eval, only: eval_ptr
   type(c_funptr), value :: ptr
   call c_f_procpointer(ptr, eval_ptr)
end subroutine gmin_set_eval
