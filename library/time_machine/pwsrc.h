#ifndef GUARD_pwsrc_h
#define GUARD_pwsrc_h

#include "petsc.h"

void srcJy(PetscScalar *Jy,
	   PetscInt ix_start, PetscInt nx_segment, PetscReal dx, PetscReal dz,
	   PetscReal kwav, PetscReal alpha,
	   PetscScalar Ay);

#endif
