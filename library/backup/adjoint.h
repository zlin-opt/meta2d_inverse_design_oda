#ifndef GUARD_adjoint_h
#define GUARD_adjoint_h

#include <assert.h>
#include "petsc.h"
#include "array2vec.h"
#include "vec.h"
#include "solver.h"
#include "varh.h"
#include "misc.h"
#include "type.h"
#include "output.h"

void strehlpieces(MPI_Comm subcomm, PetscInt Nx, PetscInt Nz, PetscScalar *dof, Mat A,
		  Mat DDe, Vec epsDiff, Vec epsBkg, PetscReal omega,
		  KSP ksp, int *its, int maxit,
		  PetscScalar *Jsrc, PetscInt iz_src,
		  Vec ge, Vec gh, Mat Pdg,
		  PetscScalar *fEy, PetscScalar *fHx, PetscScalar *nPz,
		  PetscScalar *fEy_grad, PetscScalar *fHx_grad, PetscScalar *nPz_grad);

/*
  Solves Maxwell's equation: ( DDe - omega^2 eps ) E = i omega J and 
  computes the (complex-valued) farfield = F \dot E and its (complex-valued) adjoint gradient. 
  Note that eps = epsDiff * ( A . dof ) + epsBkg. Here, the matrix A extends the (1/0) dof array to the simulation domain
  [filling in 0's for the background]. A is constructed in dof2dom.c 
  Note that the domain gradient is reduced to DOF gradient by multiplying with transpose(A).
  The Petsc solver context ksp must be setup outside (as a direct solver). It is also reused here to solve the adjoint problems.
*/

#endif
