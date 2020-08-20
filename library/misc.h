#ifndef GUARD_misc_h
#define GUARD_misc_h

#include <assert.h>
#include "petsc.h"
#include "array2vec.h"
#include "type.h"

void matmult_arrays(Mat W, void *_u, TYPE typein, void *_v, TYPE typeout, PetscInt transposeW);

PetscInt integer_sum(PetscInt *x, PetscInt i0, PetscInt i1);

PetscReal real_sum(PetscReal *x, PetscInt i0, PetscInt i1);

PetscReal find_max(PetscReal *data, PetscInt n);

#endif
