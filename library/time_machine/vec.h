#ifndef GUARD_vec_h
#define GUARD_vec_h

#include <assert.h>
#include "petsc.h"
#include "array2vec.h"
#include "type.h"

void setlayer_eps(Vec eps,
		  PetscInt Nx, PetscInt Nz,
		  PetscInt num_layers, PetscInt *zstarts, PetscInt *thickness,
		  PetscScalar *epsilon);

void vecfill_zslice(Vec v, PetscScalar *_v, PetscInt Nx, PetscInt Nz, PetscInt iz);

#endif
