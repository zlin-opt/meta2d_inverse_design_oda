#ifndef GUARD_array2vec_h
#define GUARD_array2vec_h

#include "petsc.h"
#include "type.h"

PetscErrorCode array2mpi(void *pt, TYPE typeinput, Vec v);

PetscErrorCode mpi2array(Vec v, void *pt, TYPE typeoutput, PetscInt n);

#endif
