#ifndef GUARD_maxwell_h
#define GUARD_maxwell_h

#include <assert.h>
#include "petsc.h"
#include "type.h"

PetscScalar pml_s(PetscInt N, PetscInt Npml0, PetscInt Npml1, PetscReal i, PetscReal delta, PetscReal omega);

void create_Dze(MPI_Comm comm, Mat *Dze_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz,
		PetscReal omega);
void create_Dxe(MPI_Comm comm, Mat *Dxe_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz,
		PetscReal omega);

void create_Dzh(MPI_Comm comm, Mat *Dzh_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz,
		PetscReal omega);
void create_Dxh(MPI_Comm comm, Mat *Dxh_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz,
		PetscReal omega);


void create_DDe(MPI_Comm comm, Mat *DDe_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz,
		PetscReal omega);

void syncHx(MPI_Comm comm, Mat *Ah_out, PetscInt Nx, PetscInt Nz);

#endif
