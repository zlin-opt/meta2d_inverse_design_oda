#ifndef GUARD_solver_h
#define GUARD_solver_h

#include "petsc.h"

PetscErrorCode setupKSPDirect(MPI_Comm comm, KSP *kspout, PetscInt maxit);

PetscErrorCode SolveMatrixDirect(MPI_Comm comm, KSP ksp, Mat M, Vec b, Vec x, PetscInt *its, PetscInt maxit);

#endif
