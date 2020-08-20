#ifndef GUARD_qforms_h
#define GUARD_qforms_h

#include <assert.h>
#include "petsc.h"
#include "type.h"
#include "maxwell.h"
#include "array2vec.h"
#include "farfield.h"

void create_P(MPI_Comm comm, Mat *P_out, PetscInt dagger,
	      PetscInt nx, PetscInt nz,
	      PetscInt npmlx0, PetscInt npmlx1, PetscInt npmlz0, PetscInt npmlz1,
	      PetscReal dx, PetscReal dz,
	      PetscReal omega,
	      PetscInt ix_lstart, PetscInt nx_mtr, PetscInt iz_mtr);

void gforms(MPI_Comm comm, Vec ge, Vec gh,
	    PetscInt nx, PetscInt nz,
	    PetscInt npmlx0, PetscInt npmlx1, PetscInt npmlz0, PetscInt npmlz1,
	    PetscReal dx, PetscReal dz,
	    PetscReal omega,
	    PetscInt ix_lstart, PetscInt nx_mtr, PetscInt ix_gstart, PetscInt iz_mtr,
	    PetscReal xfar, PetscReal zfar,
	    PetscReal origx, PetscReal origz);

#endif
