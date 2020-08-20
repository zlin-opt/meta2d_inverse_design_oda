#ifndef GUARD_varh_h
#define GUARD_varh_h

#include "petsc.h"
#include "type.h"

void varh_expand(PetscScalar *hdof, PetscScalar *edof, PetscInt nx, PetscInt ncells, PetscInt nlayers, PetscInt *mz, PetscReal beta, PetscInt zfixed);

void varh_contract(PetscScalar *egrad, PetscScalar *hgrad, PetscScalar *hdof, PetscInt nx, PetscInt ncells, PetscInt nlayers, PetscInt *mz, PetscReal beta, PetscInt zfixed);

#endif
