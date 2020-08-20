#ifndef GUARD_ovmat_h
#define GUARD_ovmat_h

#include "petsc.h"
#include "type.h"

void create_ovmat(MPI_Comm comm, Mat *Wout, PetscInt *mrows_per_cell, PetscInt *cell_start, PetscInt nx, PetscInt px, PetscInt ncells, PetscInt nlayers, PetscInt pmlx0, PetscInt pmlx1, PetscInt *mz, PetscInt mzslab);

#endif
