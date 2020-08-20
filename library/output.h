#ifndef GUARD_output_h
#define GUARD_output_h

#include "petsc.h"
#include "type.h"

PetscErrorCode writetofile(MPI_Comm comm, char *filename, void *data, TYPE typeout, PetscInt n);

#endif
