function gmin_revision() bind(c, name="gmin_revision") result(ptr)
   use iso_c_binding
   type(c_ptr) :: ptr
   include "svnrev.inc"
   character(kind=c_char, len=len(svnrev) + 1), target, save :: rev = svnrev // c_null_char
   ptr = c_loc(rev)
end function gmin_revision

function gmin_tarball_sha256() bind(c, name="gmin_tarball_sha256") result(ptr)
   use iso_c_binding
   type(c_ptr) :: ptr
   include "svnrev.inc"
   character(kind=c_char, len=len(tarball_sha256) + 1), target, save :: sha = tarball_sha256 // c_null_char
   ptr = c_loc(sha)
end function gmin_tarball_sha256

subroutine gmin_lbfgs_run(n, x, m, eps, itmax, maxbfgs_in, dguess_in, maxerise_in, &
      maxefall_in, mflag_out, energy, itdone, info) bind(c, name="gmin_lbfgs_run")
   use iso_c_binding
   use prec, only: real64
   use commons
   use genrigid, only: natoms
   use gmin_helpers, only: ftest
   use gmin_eval, only: harness_eps
   use mc_helpers, only: evap, evapreject
   use lbfgs_mod, only: mymylbfgs, lbfgs_destroy, cudaforlbfgs
   implicit none
   integer(c_int), value :: n, m, itmax
   real(c_double), intent(inout) :: x(n)
   real(c_double), value :: eps, maxbfgs_in, dguess_in, maxerise_in, maxefall_in
   integer(c_int), intent(out) :: mflag_out, itdone, info
   real(c_double), intent(out) :: energy
   logical :: mflag, diagco, reset
   integer :: ndone, natom
   real(real64), allocatable :: coords(:)

   info = 0
   mflag_out = 0
   itdone = 0
   energy = 0.0d0
   if (mod(n, 3) /= 0 .or. n <= 0 .or. m <= 0 .or. itmax <= 0) then
      info = 3
      return
   end if
   natom = n / 3
   natoms = natom
   which_pot = 0
   maxbfgs = maxbfgs_in
   dguess = dguess_in
   maxerise = maxerise_in
   maxefall = maxefall_in
   harness_eps = eps
   ftest = .false.
   coldfusion = .false.
   lmperr = .false.
   evap = .false.
   evapreject = .false.
   cudaforlbfgs = .false.
   if (.not. allocated(nq)) allocate(nq(1))
   nq(1) = 0
   call lbfgs_destroy()
   allocate(coords(n))
   coords(1:n) = x(1:n)
   diagco = .false.
   reset = .true.
   mflag = .false.
   ndone = 0
   call mymylbfgs(n, m, coords, diagco, eps, mflag, energy, itmax, ndone, reset, 1)
   x(1:n) = coords(1:n)
   itdone = ndone
   if (ftest) then
      info = 2
      mflag_out = 0
   else if (mflag) then
      info = 0
      mflag_out = 1
   else
      info = 1
      mflag_out = 0
   end if
end subroutine gmin_lbfgs_run
