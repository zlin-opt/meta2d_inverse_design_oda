#ifndef GUARD_farfield_h
#define GUARD_farfield_h

#include "petsc.h"
#include "array2vec.h"
#include "misc.h"
#include "type.h"

PetscScalar hk1(PetscInt n, PetscReal x);

void GEH(PetscReal  x, PetscReal  z,
	 PetscReal x0, PetscReal z0,
	 PetscReal omega, PetscReal mu, PetscReal eps,
	 PetscScalar *fEe, PetscScalar *fEh,
	 PetscScalar *fHe, PetscScalar *fHh);

#endif
