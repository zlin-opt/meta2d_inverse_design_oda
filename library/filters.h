#ifndef GUARD_filters_h
#define GUARD_filters_h

#include "petsc.h"
#include "type.h"

void density_filter(MPI_Comm comm, Mat *Qout, PetscInt Nx, PetscInt nlayers, PetscReal filt_radius, PetscReal sigma, PetscInt normalized);
//weight = exp(-dist^2/sigma^2) 

void pixbunch(MPI_Comm comm, Mat *Bout, PetscInt Nx, PetscInt nlayers, PetscInt pix_rad);

void mirrormat(MPI_Comm comm, Mat *Qout, PetscInt Nx, PetscInt nlayers);

#endif
