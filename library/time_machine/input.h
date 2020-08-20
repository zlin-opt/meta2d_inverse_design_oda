#ifndef GUARD_input_h
#define GUARD_input_h

#include "petsc.h"
#include "petscsys.h"
#include "type.h"

PetscErrorCode getreal(const char *flag, PetscReal *var, PetscReal autoval);

PetscErrorCode getint(const char *flag, PetscInt *var, PetscInt autoval);

PetscErrorCode getstr(const char *flag, char *filename, const char default_filename[]);

PetscErrorCode getintarray(const char *flag, PetscInt *z, PetscInt *nz, PetscInt default_val);

PetscErrorCode getrealarray(const char *flag, PetscReal *r, PetscInt *nr, PetscReal default_val);

PetscErrorCode readfromfile(char *name, void *data, TYPE typein, PetscInt n);

#endif
